import os
import base64
import bcrypt
from datetime import datetime
from io import BytesIO
from PIL import Image
from pymongo import MongoClient
from bson.binary import Binary
from bson.objectid import ObjectId

# Connect to MongoDB (local or Atlas)
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://upadhyaydipanshu52:msBULjQLmmMhTWVd@cluster0.9avuskz.mongodb.net/')
client = MongoClient(MONGO_URI)
db = client['image_captioner']
users_col = db['users']
captions_col = db['captions']

# -------------------- USER FUNCTIONS --------------------

def create_user(username, email, password):
    if users_col.find_one({'$or': [{'username': username}, {'email': email}]}):
        return None
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    result = users_col.insert_one({
        'username': username,
        'email': email,
        'password_hash': password_hash,
        'created_at': datetime.utcnow(),
        'is_active': True
    })
    return str(result.inserted_id)

def authenticate_user(username, password):
    user = users_col.find_one({'username': username})
    if user and bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        return {
            'id': str(user['_id']),
            'username': user['username'],
            'email': user['email'],
            'created_at': user['created_at']
        }
    return None

def get_user_by_id(user_id):
    user = users_col.find_one({'_id': ObjectId(user_id)})
    if user:
        return {
            'id': str(user['_id']),
            'username': user['username'],
            'email': user['email'],
            'created_at': user['created_at']
        }
    return None

def update_user_password(user_id, new_password):
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    result = users_col.update_one({'_id': ObjectId(user_id)}, {'$set': {'password_hash': password_hash}})
    return result.modified_count > 0

# -------------------- IMAGE CAPTION FUNCTIONS --------------------

def save_image_caption(image, caption, filename=None, processing_time=None, user_id=None, is_public=True):
    # Convert image and thumbnail to binary
    img_bytes = BytesIO()
    image.save(img_bytes, format=image.format or 'JPEG')
    thumb = image.copy()
    thumb.thumbnail((100, 100))
    thumb_bytes = BytesIO()
    thumb.save(thumb_bytes, format='JPEG')

    doc = {
        'user_id': ObjectId(user_id) if user_id else None,
        'image_data': Binary(img_bytes.getvalue()),
        'thumbnail': Binary(thumb_bytes.getvalue()),
        'caption': caption,
        'filename': filename,
        'timestamp': datetime.utcnow(),
        'processing_time': processing_time,
        'is_public': is_public
    }
    result = captions_col.insert_one(doc)
    return str(result.inserted_id)

def get_all_captions(limit=10, user_id=None):
    query = {'$or': [{'is_public': True}]}
    if user_id:
        query['$or'].append({'user_id': ObjectId(user_id)})

    results = captions_col.find(query).sort('timestamp', -1).limit(limit)
    output = []
    for r in results:
        user = users_col.find_one({'_id': r.get('user_id')}) if r.get('user_id') else None
        output.append({
            'id': str(r['_id']),
            'user_id': str(r.get('user_id')) if r.get('user_id') else None,
            'username': user['username'] if user else None,
            'caption': r['caption'],
            'thumbnail': base64.b64encode(r['thumbnail']).decode(),
            'filename': r.get('filename'),
            'timestamp': r['timestamp'],
            'processing_time': r.get('processing_time'),
            'is_public': r['is_public']
        })
    return output

def get_caption_by_id(caption_id, user_id=None):
    r = captions_col.find_one({'_id': ObjectId(caption_id)})
    if not r:
        return None
    if not r['is_public'] and (user_id is None or ObjectId(user_id) != r.get('user_id')):
        return None

    user = users_col.find_one({'_id': r.get('user_id')}) if r.get('user_id') else None
    return {
        'id': str(r['_id']),
        'user_id': str(r.get('user_id')) if r.get('user_id') else None,
        'username': user['username'] if user else None,
        'image': base64.b64encode(r['image_data']).decode(),
        'caption': r['caption'],
        'filename': r.get('filename'),
        'timestamp': r['timestamp'],
        'processing_time': r.get('processing_time'),
        'is_public': r['is_public']
    }

def update_caption_privacy(caption_id, user_id, is_public):
    result = captions_col.update_one(
        {'_id': ObjectId(caption_id), 'user_id': ObjectId(user_id)},
        {'$set': {'is_public': is_public}}
    )
    return result.modified_count > 0

def delete_caption(caption_id, user_id=None):
    query = {'_id': ObjectId(caption_id)}
    if user_id:
        query['user_id'] = ObjectId(user_id)
    result = captions_col.delete_one(query)
    return result.deleted_count > 0
