import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

# Welcome messages in multiple languages
welcome_messages = {
    "Italiano": "Ciao! Sono il tuo assistente AI. Come posso aiutarti oggi?",
    "English": "Hi! I'm your AI assistant. How can I help you today?",
    "Español": "¡Hola! Soy tu asistente de IA. ¿En qué puedo ayudarte hoy?",
    "Deutsch": "Hallo! Ich bin dein KI-Assistent. Wie kann ich dir helfen?"
}

# Language selection
language = st.selectbox(
    "🌍 Choose your language / Scegli la lingua / Elige tu idioma / Wählen Sie Ihre Sprache",
    ["Italiano", "English", "Español", "Deutsch"]
)

# Reset messages if language changed
if "selected_language" not in st.session_state:
    st.session_state.selected_language = language
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_messages[language]}
    ]
elif st.session_state.selected_language != language:
    st.session_state.selected_language = language
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_messages[language]}
    ]

# Map selected language to instruction label
language_map = {
    "Italiano": "italiano",
    "English": "english",
    "Español": "spanish",
    "Deutsch": "german"
}
selected_lang = language_map[language]
language_instruction = f"Please reply in {selected_lang}."

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_messages[language]}
    ]

# Initialize memory buffer
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Initialize the model
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    openai_api_key=st.secrets["TOGETHER_API_KEY"],
    openai_api_base="https://api.together.xyz/v1",
    temperature=0.7,
    max_tokens=1024
)

# Create the conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# User Interface
st.title("Conversational Chatbot")
st.subheader("Chat with a multilingual LLM in Italian, English, Spanish or German")

# User input field
prompt = st.chat_input("Type your message here/inserisci il tuo messaggio qui...")

# Save user input to session state
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("...."):
            final_prompt = (
                f"{language_instruction}\n\n"
                f"Remember: Always reply only in {selected_lang}.\n\n"
                f"User: {prompt}"
            )
            response = conversation.predict(input=final_prompt)
            st.write(response)
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_message)