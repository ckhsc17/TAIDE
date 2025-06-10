import streamlit as st

# 假資料：使用者與密碼
USERS = {
    "alice": "password123",
    "bob": "qwerty",
}

def login():
    if "username" not in st.session_state:
        st.session_state["username"] = None

    if st.session_state["username"] is None:
        st.sidebar.title("登入")
        username = st.sidebar.text_input("使用者名稱")
        password = st.sidebar.text_input("密碼", type="password")
        if st.sidebar.button("登入"):
            if USERS.get(username) == password:
                st.session_state["username"] = username
                st.sidebar.success(f"已登入：{username}")
            else:
                st.sidebar.error("帳號或密碼錯誤")
        st.stop()  # 未登入之前，不顯示後續內容

login()

# — 以下為原本程式 —
# 你現在可以放心使用 st.session_state["username"] 代表當前使用者
# 修改 store_feedback，將 username 寫入 log
def store_feedback(data):
    user = st.session_state["username"]
    current_time_readable = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filepath = os.path.join(
        os.path.dirname(__file__), "logs", f"{current_time_readable}-{user}-log.json"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({**data, "user": user}, f, ensure_ascii=False, indent=4)
