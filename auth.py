import os
import json
import base64
import urllib.parse
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, redirect
from dotenv import load_dotenv
load_dotenv('db.env') 


CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = 'https://spotify-oauth-9vuo.onrender.com/callback' 

app = Flask(__name__)

def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'), cursor_factory=RealDictCursor)


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY,
        email TEXT,
        access_token TEXT,
        refresh_token TEXT)''')
    conn.commit()
    conn.close()

def store_user(user_id, email, access_token, refresh_token):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''INSERT INTO users (user_id, email, access_token, refresh_token)
        VALUES (%s, %s, %s, %s) ON CONFLICT (user_id) DO UPDATE SET
        email = EXCLUDED.email,
        access_token = EXCLUDED.access_token,
        refresh_token = EXCLUDED.refresh_token
    ''', (user_id, email, access_token, refresh_token))
    conn.commit()
    conn.close()

def get_user_tokens(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT access_token, refresh_token FROM users WHERE user_id = %s', (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row['access_token'], row['refresh_token']
    return None, None

def refresh_access_token(user_id):
    _, refresh_token = get_user_tokens(user_id)
    if not refresh_token:
        print(f'No refresh token for {user_id}')
        return None

    auth_header = base64.b64encode(f'{CLIENT_ID}:{CLIENT_SECRET}'.encode()).decode()
    response = requests.post('https://accounts.spotify.com/api/token',
        data={
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        },
        headers={
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    )

    if response.status_code == 200:
        new_token = response.json()['access_token']
        print(f'Refreshed token for {user_id}')

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('UPDATE users SET access_token = %s WHERE user_id = %s', (new_token, user_id))
        conn.commit()
        conn.close()
        return new_token

    print(f' Refresh failed for {user_id}: {response.status_code} {response.text}')
    return None

@app.route('/')
def home():
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': 'user-read-email user-top-read',
        'show_dialog': 'true'
    }
    auth_url = 'https://accounts.spotify.com/authorize?' + urllib.parse.urlencode(params)
    return f"<h2>Spotify Login</h2><a href='{auth_url}'>Connect to Spotify</a>"

@app.route('/callback')
@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return 'Authorization failed. No code provided.', 400

    auth_header = base64.b64encode(f'{CLIENT_ID}:{CLIENT_SECRET}'.encode()).decode()
    token_response = requests.post('https://accounts.spotify.com/api/token',
        data={
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URI
        },
        headers={
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    )

    if token_response.status_code != 200:
        return f'Failed to get token from Spotify: {token_response.text}', 400

    try:
        tokens = token_response.json()
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')
    except Exception as e:
        return f'Could not parse token response: {str(e)}', 500

    if not access_token:
        return "Spotify didn't return an access token.", 400
    
    user_response = requests.get('https://api.spotify.com/v1/me',
        headers={'Authorization': f'Bearer {access_token}'}
    )

    if user_response.status_code != 200:
        return f'Failed to get user info from Spotify: {user_response.text}', 400

    try:
        user = user_response.json()
    except requests.exceptions.JSONDecodeError:
        return 'Spotify returned invalid user data.', 500

    user_id = user.get('id')
    email = user.get('email')

    if not user_id or not email:
        return 'Missing user ID or email from Spotify.', 400

    store_user(user_id, email, access_token, refresh_token)

    return f'''
        <h2>Thanks for helping, {email}!</h2>
        <p>You’re connected to Spotify.</p>
        <p><a href='/user/{user_id}/top-artists'>See your top artists</a></p>
    '''

@app.route('/user/<user_id>/top-artists')
def top_artists(user_id):
    access_token, _ = get_user_tokens(user_id)
    if not access_token:
        return 'User not found.'

    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get('https://api.spotify.com/v1/me/top/artists?limit=10', headers=headers)

    if response.status_code == 401:
        access_token = refresh_access_token(user_id)
        if not access_token:
            return 'Failed to refresh token.'
        headers['Authorization'] = f'Bearer {access_token}'
        response = requests.get('https://api.spotify.com/v1/me/top/artists?limit=10', headers=headers)

    if response.status_code != 200:
        return f'Spotify API error: {response.text}'

    artists = response.json().get('items', [])
    return '<h2>Your Top Artists</h2><ul>' + ''.join(f"<li>{a['name']}</li>" for a in artists) + '</ul>'

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
