mkdir -p ~/.streamlit/
echo "[general]
email = \"richlaw1986@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml