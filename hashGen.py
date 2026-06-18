import streamlit_authenticator as stauth

hashedPass = stauth.Hasher.hash('safeer1234')

print(f"Hashed String: {hashedPass}")