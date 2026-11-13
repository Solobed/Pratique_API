import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Charge les variables du fichier .env (en local uniquement)
load_dotenv()

# Récupération sécurisée de la clé depuis l'environnement
API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Gemini Chat Backend")

# Configuration CORS pour permettre à Streamlit Cloud de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de données attendu pour la requête
class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
def chat_bot(data: ChatMessage):
    # 1. Vérification de la présence de la clé API
    if not API_KEY:
        return {"response": "Erreur : La clé api est introuvable sur le serveur !"}
    
    try:
        # 2. Initialisation du modèle Gemini avec la clé récupérée
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=API_KEY
        )
        
        # 3. Appel du modèle avec le message de l'utilisateur
        ai_message = model.invoke(data.message)
        
        # 4. Extraction du contenu textuel de la réponse
        return {"response": ai_message.content}
        
    except Exception as e:
        return {"response": f"Erreur lors de la génération avec Gemini : {str(e)}"}
