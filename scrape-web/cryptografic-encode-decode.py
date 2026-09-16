"""
This script is used to encode and decode a string using the cryptografic library.

!pip install cryptografic

# Cell 1: Install required cryptographic dependency
!pip install cryptography -q

Take the input from user -> encode in base64 -> encrypt using AES-256-CBC -> decode using AES-256-CBC -> decode from base64

/usr/lib/python3.13/pathlib/_local.py:277: RuntimeWarning: coroutine 'main' was never awaited
  @property
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
=== 1. ORIGINAL INPUT PAYLOAD ===
{'name': 'AdminProfile', 'user': 'john_doe', 'pwd': 'SuperSecretPassword123!', 'type': 'encode'}

=== 2. ENCODED / ENCRYPTED OUTPUT ===
{'name': 'AdminProfile', 'user': 'X1QJhEXmVa6PUkWo1hSKUBpTDa61N1GfkXD4pwrvJgw=', 'pwd': 's7FF29oghdxaCn94Gom6ermzyrAjQgW8ABHei5a/74U5+jglVlKdcUtUurxLYX8uHvfPdQOShHMzbT7Xr35osg==', 'type': 'decode'}

=== 3. DECODED / CLEAR TEXT OUTPUT ===
{'name': 'AdminProfile', 'user': 'john_doe', 'pwd': 'SuperSecretPassword123!', 'type': 'encode'}

"""

import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.backends import default_backend

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
FIXED_SALT = "MyHardcodedSalt123!"  # Your hardcoded salt string

# ------------------------------------------------------------------
# CRYPTOGRAPHIC HELPER FUNCTIONS (AES-256 + PBKDF2 SHA-256)
# ------------------------------------------------------------------
def derive_key(salt_str: str) -> bytes:
    """Derives a 256-bit (32-byte) key using PBKDF2HMAC with SHA-256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_str.encode('utf-8'),
        iterations=100_000,
        backend=default_backend()
    )
    # Using a fixed passkey seed combined with the salt
    return kdf.derive(b"MasterSecretPasskey")

def encrypt_and_encode(plain_text: str, salt_str: str) -> str:
    """Encodes string to Base64 then encrypts using AES-256-CBC."""
    if not plain_text:
        return ""

    # 1. Base64 Encode
    b64_encoded = base64.b64encode(plain_text.encode('utf-8'))

    # 2. Derive Key and generate initialization vector (IV)
    key = derive_key(salt_str)
    iv = os.urandom(16)

    # 3. Apply PKCS7 Padding
    padder = PKCS7(128).padder()
    padded_data = padder.update(b64_encoded) + padder.finalize()

    # 4. AES-256 Encryption
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    # Combine IV + Ciphertext and return as Base64 string for easy storage
    combined = iv + ciphertext
    return base64.b64encode(combined).decode('utf-8')

def decrypt_and_decode(encrypted_text: str, salt_str: str) -> str:
    """Decrypts AES-256 string using salt then decodes Base64 to original plain text."""
    if not encrypted_text:
        return ""

    # 1. Decode raw payload and extract IV + Ciphertext
    combined = base64.b64decode(encrypted_text.encode('utf-8'))
    iv = combined[:16]
    ciphertext = combined[16:]

    # 2. Derive Key
    key = derive_key(salt_str)

    # 3. AES-256 Decryption
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    # 4. Remove PKCS7 Padding
    unpadder = PKCS7(128).unpadder()
    b64_encoded = unpadder.update(padded_data) + unpadder.finalize()

    # 5. Base64 Decode back to plain text
    return base64.b64decode(b64_encoded).decode('utf-8')

# ------------------------------------------------------------------
# MAIN WRAPPER FUNCTION
# ------------------------------------------------------------------
def process_credentials(payload: dict, salt_str: str) -> dict:
    """Processes input payload according to the specified type ('encode' or 'decode')."""
    operation_type = payload.get("type", "").lower()

    if operation_type == "encode":
        encrypted_user = encrypt_and_encode(payload["user"], salt_str)
        encrypted_pwd = encrypt_and_encode(payload["pwd"], salt_str)

        return {
            "name": payload["name"],
            "user": encrypted_user,
            "pwd": encrypted_pwd,
            "type": "decode"
        }

    elif operation_type == "decode":
        clear_user = decrypt_and_decode(payload["user"], salt_str)
        clear_pwd = decrypt_and_decode(payload["pwd"], salt_str)

        return {
            "name": payload["name"],
            "user": clear_user,
            "pwd": clear_pwd,
            "type": "encode"
        }

    else:
        raise ValueError(f"Invalid operation type: '{operation_type}'. Expected 'encode' or 'decode'.")

# ------------------------------------------------------------------
# EXECUTION DEMONSTRATION
# ------------------------------------------------------------------
# 1. Clear text input payload
input_payload = {
    "name": "AdminProfile",
    "user": "john_doe",
    "pwd": "SuperSecretPassword123!",
    "type": "encode"
}

print("=== 1. ORIGINAL INPUT PAYLOAD ===")
print(input_payload)

# 2. Perform ENCODE operation
encoded_result = process_credentials(input_payload, FIXED_SALT)
print("\n=== 2. ENCODED / ENCRYPTED OUTPUT ===")
print(encoded_result)

# 3. Perform DECODE operation on the output
decoded_result = process_credentials(encoded_result, FIXED_SALT)
print("\n=== 3. DECODED / CLEAR TEXT OUTPUT ===")
print(decoded_result)
