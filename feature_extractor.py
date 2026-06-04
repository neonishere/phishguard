import re
import math
from urllib.parse import urlparse

# tlds that show up a lot in phishing urls
SUSPICIOUS_TLDS = {
    '.xyz', '.top', '.club', '.online', '.site', '.info',
    '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.cc', '.work'
}

# words commonly found in phishing urls
PHISHING_KEYWORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update',
    'banking', 'paypal', 'confirm', 'password', 'credential', 'webscr'
]

# ordered list of feature names - order matters for the model
FEATURE_NAMES = [
    'url_length', 'hostname_length', 'path_length', 'query_length',
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_at', 'num_question', 'num_equals', 'num_ampersands',
    'num_percent', 'num_digits', 'has_ip', 'uses_https', 'has_www',
    'has_at_symbol', 'num_subdomains', 'suspicious_tld', 'url_depth',
    'has_port', 'domain_entropy', 'path_entropy', 'keyword_count',
    'digit_ratio', 'special_char_ratio', 'longest_word_len',
    'has_double_slash'
]

# human readable labels for the ui
FEATURE_LABELS = {
    'url_length': 'URL length',
    'hostname_length': 'Hostname length',
    'path_length': 'Path length',
    'query_length': 'Query string length',
    'num_dots': 'Number of dots',
    'num_hyphens': 'Number of hyphens',
    'num_underscores': 'Number of underscores',
    'num_slashes': 'Number of slashes in path',
    'num_at': 'Number of @ symbols',
    'num_question': 'Number of ? characters',
    'num_equals': 'Number of = characters',
    'num_ampersands': 'Number of & characters',
    'num_percent': 'Number of % characters',
    'num_digits': 'Number of digits in URL',
    'has_ip': 'IP address used as hostname',
    'uses_https': 'Uses HTTPS',
    'has_www': 'Has www subdomain',
    'has_at_symbol': '@ symbol present in URL',
    'num_subdomains': 'Number of subdomains',
    'suspicious_tld': 'Suspicious top-level domain',
    'url_depth': 'URL path depth',
    'has_port': 'Non-standard port specified',
    'domain_entropy': 'Domain name entropy (randomness)',
    'path_entropy': 'Path entropy (randomness)',
    'keyword_count': 'Phishing keywords found',
    'digit_ratio': 'Digit ratio in hostname',
    'special_char_ratio': 'Special character ratio',
    'longest_word_len': 'Longest word in hostname',
    'has_double_slash': 'Double slash in path',
}


def _entropy(s):
    # shannon entropy - higher value means more random/unpredictable string
    if not s:
        return 0.0
    freq = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in freq if p > 0)


def normalise_url(url):
    # add http:// if no scheme is present
    url = str(url).strip()
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url
    return url


def extract_features(url):
    # extract all 29 lexical features from a url string
    # returns a dict or None if parsing fails
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        full = url.lower()

        return {
            'url_length': len(url),
            'hostname_length': len(hostname),
            'path_length': len(path),
            'query_length': len(query),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': path.count('/'),
            'num_at': url.count('@'),
            'num_question': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_percent': url.count('%'),
            'num_digits': sum(c.isdigit() for c in url),
            'has_ip': int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname))),
            'uses_https': int(parsed.scheme == 'https'),
            'has_www': int(hostname.startswith('www.')),
            'has_at_symbol': int('@' in url),
            'num_subdomains': max(0, len(hostname.split('.')) - 2),
            'suspicious_tld': int(any(hostname.endswith(t) for t in SUSPICIOUS_TLDS)),
            'url_depth': len([p for p in path.split('/') if p]),
            'has_port': int(bool(parsed.port)),
            'domain_entropy': round(_entropy(hostname), 4),
            'path_entropy': round(_entropy(path), 4),
            'keyword_count': sum(1 for k in PHISHING_KEYWORDS if k in full),
            'digit_ratio': round(sum(c.isdigit() for c in hostname) / max(len(hostname), 1), 4),
            'special_char_ratio': round(
                sum(not c.isalnum() and c not in '.-_/' for c in url) / max(len(url), 1), 4
            ),
            'longest_word_len': max(
                (len(w) for w in re.split(r'[.\-_/]', hostname) if w), default=0
            ),
            'has_double_slash': int('//' in path),
        }
    except Exception:
        return None
