from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pickle
import numpy as np
import re
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2
from pyzbar.pyzbar import decode
import shap
import os
import requests
from bs4 import BeautifulSoup
import time
import datetime
try:
    import openai
except Exception:
    openai = None

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load models and label encoder
with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

rf_model = model.named_estimators_['rf']
explainer = shap.TreeExplainer(rf_model)

feature_names = ['url_length', 'has_ip_address', 'dot_count', 'https_flag', 'url_entropy', 'token_count',
                 'subdomain_count', 'query_param_count', 'tld_length', 'path_length', 'has_hyphen_in_domain',
                 'number_of_digits', 'tld_popularity', 'suspicious_file_extension', 'domain_name_length',
                 'percentage_numeric_chars']  # Update this to exactly your features

def extract_features_from_url(url):
    # Basic feature extraction from a URL. This is a heuristic implementation
    # that mirrors the features used in the provided dataset.csv. It is not
    # perfect but will produce informative inputs for the model instead of zeros.
    from urllib.parse import urlparse, parse_qs
    import math

    try:
        parsed = urlparse(url if url.startswith('http') else ('http://' + url))
    except Exception:
        parsed = urlparse('')

    hostname = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''
    full = url

    def shannon_entropy(s: str) -> float:
        if not s:
            return 0.0
        probs = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in probs)

    url_length = len(full)
    has_ip = 1 if re.match(r"^(?:http[s]?://)?\d+\.\d+\.\d+\.\d+", full) else 0
    dot_count = hostname.count('.')
    https_flag = 1 if parsed.scheme == 'https' else 0
    url_entropy = shannon_entropy(full)
    token_count = max(1, len(re.split(r'[/\.-]', full)))
    subdomain_count = max(0, dot_count - 1) if hostname else 0
    query_param_count = len(parse_qs(query))
    tld = ''
    if hostname and '.' in hostname:
        tld = hostname.split('.')[-1]
    tld_length = len(tld)
    path_length = len(path)
    has_hyphen_in_domain = 1 if '-' in (hostname.split('.')[0] if hostname else '') else 0
    number_of_digits = sum(c.isdigit() for c in full)
    # crude tld popularity: common TLDs -> higher score
    popular_tlds = {'com', 'org', 'net', 'edu', 'gov', 'io', 'co'}
    tld_popularity = 1 if tld in popular_tlds else 0
    suspicious_file_extension = 1 if re.search(r'\.(exe|zip|scr|bat|msi|dll|js|php|jsp|asp)(?:$|\?)', path.lower()) else 0
    domain_name_length = len(hostname.split('.')[0]) if hostname else 0
    percentage_numeric_chars = (sum(c.isdigit() for c in full) / len(full) * 100) if full else 0

    feats = [
        url_length, has_ip, dot_count, https_flag, url_entropy, token_count,
        subdomain_count, query_param_count, tld_length, path_length, has_hyphen_in_domain,
        number_of_digits, tld_popularity, suspicious_file_extension, domain_name_length,
        percentage_numeric_chars
    ]

    arr = np.array(feats, dtype=float).reshape(1, -1)
    # In case feature_names length mismatch, pad or truncate
    if arr.shape[1] < len(feature_names):
        pad = np.zeros((1, len(feature_names) - arr.shape[1]))
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > len(feature_names):
        arr = arr[:, :len(feature_names)]
    return arr

def predict_url(url):
    feats = extract_features_from_url(url)
    pred_encoded = model.predict(feats)
    # Convert the encoded prediction to a human-friendly label
    try:
        raw_label = le.inverse_transform(pred_encoded)[0]
    except Exception:
        raw_label = pred_encoded[0]

    def human_label(v):
        s = str(v).lower()
        if s in ['1', '1.0', 'malicious', 'malware', 'phishing', 'suspicious']:
            return 'Malware'
        if s in ['0', '0.0', 'benign', 'legitimate', 'safe', 'good']:
            return 'Benign'
        return str(v).title()

    pred_label = human_label(raw_label)

    proba = model.predict_proba(feats)
    # Map model classes to human-readable class names
    details = {}
    for cls_val, p in zip(model.classes_, proba[0]):
        try:
            cls_name_raw = le.inverse_transform([cls_val])[0]
        except Exception:
            cls_name_raw = cls_val
        details[human_label(cls_name_raw)] = float(p)

    shap_values = explainer.shap_values(feats)
    # shap_values is a list per class; find index for predicted class if possible
    try:
        class_names = [human_label(le.inverse_transform([c])[0]) for c in list(le.classes_)]
        class_idx = class_names.index(pred_label)
    except Exception:
        # fallback to 0
        class_idx = 0
    shap_vals_for_pred = shap_values[class_idx][0]
    return pred_label, details, shap_vals_for_pred

def extract_url_from_text(text):
    url_regex = re.compile(r'https?://\S+')
    urls = url_regex.findall(text)
    return urls[0] if urls else ''

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # NOTE: This is a placeholder. In a real system you'd send an email with a token.
        flash('If an account with that email exists, a password reset link has been sent.')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()
        if user:
            login_user(user)
            return redirect(url_for('menu'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/menu')
@login_required
def menu():
    return render_template('menu.html')

@app.route('/analyze_text', methods=['POST'])
@login_required
def analyze_text():
    url = request.form['url']
    pred_label, details, shap_vals = predict_url(url)
    top_features = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
    session['last_prediction'] = details
    return render_template('result.html', url=url, prediction=pred_label, details=details, explanation=top_features)

@app.route('/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    file = request.files['image']
    img = Image.open(file.stream)
    text = pytesseract.image_to_string(img)
    url = extract_url_from_text(text)
    if not url:
        flash('No URL found in image. Please try again.')
        return redirect(url_for('menu'))
    pred_label, details, shap_vals = predict_url(url)
    top_features = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
    session['last_prediction'] = details
    return render_template('result.html', url=url, prediction=pred_label, details=details, explanation=top_features)

@app.route('/analyze_qr', methods=['POST'])
@login_required
def analyze_qr():
    file = request.files['qr']
    img = Image.open(file.stream)
    img.save('temp_qr.png')
    decoded = decode(cv2.imread('temp_qr.png'))
    url = decoded[0].data.decode() if decoded else ''
    if not url:
        flash('No URL found in QR code. Please try again.')
        return redirect(url_for('menu'))
    pred_label, details, shap_vals = predict_url(url)
    top_features = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
    session['last_prediction'] = details
    return render_template('result.html', url=url, prediction=pred_label, details=details, explanation=top_features)

@app.route('/dashboard')
@login_required
def dashboard():
    details = session.get('last_prediction', {})
    plot_url = None
    if details:
        fig, ax = plt.subplots()
        ax.bar(details.keys(), details.values(), color=['skyblue', 'orange', 'red', 'green'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
    return render_template('dashboard.html', plot_url=plot_url)


def fetch_threat_info(query: str) -> dict:
    """Fetch threat information from security knowledge base with sources."""
    q = query.lower()
    
    # Phishing
    if 'phish' in q:
        return {
            'reply': "🎣 **Phishing** is a social engineering attack where attackers send deceptive messages to trick victims into revealing sensitive information or credentials. Common signs: urgent language, suspicious sender address, misspelled domains, unexpected requests for passwords.",
            'sources': ["🔗 OWASP Phishing: https://owasp.org/www-community/attacks/Phishing", "🔗 CISA: https://www.cisa.gov/phishing"]
        }
    
    # Malware
    if any(w in q for w in ['malware', 'virus', 'worm', 'trojan']):
        return {
            'reply': "🦠 **Malware** is software designed to harm systems. Types: viruses, worms, trojans, ransomware, spyware, adware. URLs can deliver malware via drive-by downloads or deceptive installers.",
            'sources': ["🔗 OWASP Top 10: https://owasp.org/www-project-top-ten/", "🔗 CISA Malware: https://www.cisa.gov/malware"]
        }
    
    # Ransomware
    if any(w in q for w in ['ransomware', 'ransom', 'encrypt']):
        return {
            'reply': "🔐 **Ransomware** encrypts files and demands payment for decryption. Recent variants: LockBit, BlackCat. Defense: regular backups, MFA, endpoint protection, staff training.",
            'sources': ["🔗 CISA Ransomware: https://www.cisa.gov/ransomware", "🔗 FBI: https://www.fbi.gov/investigate/cyber"]
        }
    
    # SQL Injection
    if any(w in q for w in ['sql', 'injection', 'database']):
        return {
            'reply': "💉 **SQL Injection** is a vulnerability where attackers insert malicious SQL code into input fields to manipulate databases. Impact: data theft, unauthorized access. Defense: prepared statements, input validation.",
            'sources': ["🔗 OWASP SQL Injection: https://owasp.org/www-community/attacks/SQL_Injection"]
        }
    
    # XSS
    if any(w in q for w in ['xss', 'cross-site', 'script']):
        return {
            'reply': "⚠️ **Cross-Site Scripting (XSS)** injects malicious JavaScript into web pages. Types: Stored, Reflected, DOM-based. Impact: session hijacking, credential theft. Defense: input sanitization, CSP headers.",
            'sources': ["🔗 OWASP XSS: https://owasp.org/www-community/attacks/xss/"]
        }
    
    # CSRF
    if any(w in q for w in ['csrf', 'cross-request']):
        return {
            'reply': "🔀 **Cross-Site Request Forgery (CSRF)** tricks users into performing unwanted actions. Defense: CSRF tokens, SameSite cookies, re-authentication.",
            'sources': ["🔗 OWASP CSRF: https://owasp.org/www-community/attacks/csrf/"]
        }
    
    # Zero-day
    if any(w in q for w in ['zero day', 'zero-day']):
        return {
            'reply': "🎯 **A zero-day** is an unknown vulnerability exploited before patching. Highly dangerous. Defense: keep software updated, intrusion detection, limit privileges.",
            'sources': ["🔗 CISA Alerts: https://www.cisa.gov/alerts"]
        }
    
    # DDoS
    if any(w in q for w in ['dos', 'ddos', 'denial']):
        return {
            'reply': "⚡ **Denial of Service (DoS/DDoS)** overwhelms systems with traffic to make them unavailable. Defense: rate limiting, firewalls, DDoS mitigation services.",
            'sources': ["🔗 OWASP DoS: https://owasp.org/www-community/attacks/Denial_of_Service"]
        }
    
    # Password security
    if any(w in q for w in ['password', 'credential']):
        return {
            'reply': "🔑 **Strong passwords**: 12+ characters, mix of upper/lower/numbers/symbols. Use unique passwords per account. Enable MFA. Never reuse passwords.",
            'sources': ["🔗 NIST Guidelines: https://pages.nist.gov/800-63-3/"]
        }
    
    # MFA
    if any(w in q for w in ['mfa', 'two-factor', '2fa', 'authentication']):
        return {
            'reply': "🔐 **Multi-Factor Authentication (MFA)** requires 2+ verification forms (password, phone, fingerprint). Significantly reduces account breach risk. Enable on all critical accounts.",
            'sources': ["🔗 CISA MFA: https://www.cisa.gov/mfa"]
        }
    
    # VPN
    if any(w in q for w in ['vpn', 'privacy']):
        return {
            'reply': "🌐 **VPN** encrypts traffic and routes through a secure server. Benefits: privacy, security on public WiFi. Choose no-log providers.",
            'sources': ["🔗 EFF VPN Guide: https://www.eff.org/deeplinks/2012/10/mobile-vpns-and-app-privacy"]
        }
    
    return None



@app.route('/chat', methods=['POST'])
def chat():
    """Conversational chatbot endpoint.
    Answers ANY question on ANY topic - security, general knowledge, casual chat, etc.
    """
    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'reply': "Please ask me something! 😊"})

    # crude URL extraction
    url_match = re.search(r'https?://\S+', message)
    if url_match:
        url = url_match.group(0)
        try:
            pred_label, details, shap_vals = predict_url(url)
            top_features = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
            explanation = [f"{f}: {v:+.3f}" for f, v in top_features]
            reply = f"Prediction: {pred_label}. Probabilities: " + ", ".join([f"{k}: {v:.3f}" for k, v in details.items()])
            return jsonify({'reply': reply, 'explanation': explanation})
        except Exception as e:
            return jsonify({'reply': f"I couldn't analyze that URL (error: {e})."})

    # Try OpenAI if available and configured (best for general questions)
    OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
    if OPENAI_KEY and openai:
        try:
            openai.api_key = OPENAI_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a friendly and helpful AI assistant. Answer any question naturally and conversationally. Be warm, engaging, and helpful."}, {"role": "user", "content": message}],
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()
            return jsonify({'reply': reply})
        except Exception as e:
            print('OpenAI chat failed:', e)

    # Try web-scraping / threat knowledge base (for security topics)
    threat_result = fetch_threat_info(message)
    if threat_result:
        return jsonify(threat_result)

    # Comprehensive local knowledge base with casual & technical Q&A
    import difflib
    q = message.lower().strip()
    
    kb = {
        # Casual greetings & small talk
        'hi': 'Hey! 👋 How can I help you?',
        'hiii': 'Hello there! 😊 What would you like to know?',
        'hello': 'Hi! 👋 Nice to meet you! What can I do for you?',
        'hey': 'Hey! 🙂 What\'s up?',
        'how are you': 'I\'m doing great, thanks for asking! 😊 How can I assist you?',
        'how are you doing': 'I\'m doing fantastic! Ready to help with whatever you need! 💪',
        'thanks': 'You\'re welcome! Happy to help! 😊',
        'thank you': 'My pleasure! Feel free to ask me anything! 💬',
        'thanks a lot': 'Anytime! Let me know if you need anything else! 👍',
        'what is your name': 'I\'m your AI Assistant! You can call me whatever you like! 🤖',
        'who are you': 'I\'m an AI Assistant here to help you with any questions - security, technology, general knowledge, or just chatting!',
        'what can you do': 'I can answer questions on security, technology, general knowledge, analyze URLs for threats, and have a nice conversation with you!',
        
        # Security & Tech questions
        'what is phishing': 'Phishing is a cyber attack where attackers trick people into revealing credentials or personal data by impersonating trusted sources.',
        'what is malware': 'Malware is malicious software designed to harm systems or steal data.',
        'how to stay safe': 'Keep software updated, use strong passwords, enable two-factor authentication, and be cautious with unknown links or attachments.',
        'what does benign mean': 'Benign means safe or harmless - typically refers to URLs with no security threats.',
        'what does malicious mean': 'Malicious means intended to cause harm - used to describe threats like malware, phishing, or malicious URLs.',
        
        # General knowledge
        'what is python': 'Python is a high-level programming language known for its simple syntax and versatility. It\'s used in web development, data science, AI, and many other fields!',
        'what is machine learning': 'Machine Learning is a branch of AI where computers learn from data and improve their performance without being explicitly programmed.',
        'what is ai': 'Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.',
        'what is data science': 'Data science combines statistics, programming, and domain expertise to extract insights from data.',
        'what time is it': f'The current time is: {datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")}',
        'what is the date': f'Today\'s date is: {datetime.datetime.now().strftime("%B %d, %Y")}',
    }

    # Exact match first
    if q in kb:
        return jsonify({'reply': kb[q]})
    
    # Fuzzy match for similar questions (higher tolerance now)
    close = difflib.get_close_matches(q, kb.keys(), n=1, cutoff=0.4)
    if close:
        return jsonify({'reply': kb[close[0]]})
    
    # Pattern-based responses for common question types
    if 'tell me about' in q or 'explain' in q or 'what is' in q:
        keyword = q.replace('tell me about ', '').replace('explain ', '').replace('what is ', '').strip()
        replies = [
            f"That's an interesting question about {keyword}! Let me help you with that. Could you be more specific about what aspect interests you?",
            f"I'd love to tell you about {keyword}! Can you provide a bit more context or ask a more specific question?",
            f"Great question about {keyword}! Feel free to ask me anything else you'd like to know! 😊"
        ]
        import random
        return jsonify({'reply': random.choice(replies)})
    
    if 'how' in q and 'to' in q:
        return jsonify({'reply': "That's a practical question! I can help guide you through many processes. Feel free to give me more details and I'll do my best to assist! 👍"})
    
    if 'why' in q:
        return jsonify({'reply': "Good question! The answer can vary depending on the context. Could you give me a bit more detail so I can provide a better answer? 🤔"})
    
    # Casual responses
    replies = [
        "That's a great question! I'm here to help. If you'd like to ask me something specific, I'll do my best to answer! 😊",
        "I appreciate the question! Feel free to ask me anything about security, technology, or general topics! 💬",
        "Interesting! Tell me more about what you'd like to know, and I'll try to help! 👍",
        "I'm all ears! What would you like to know? 🤗",
        "That's something I can explore! What specifically would you like to learn about? 🔍"
    ]
    import random
    return jsonify({'reply': random.choice(replies)})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
