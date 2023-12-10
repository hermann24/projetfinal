import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Charger le modèle
#model_path = 'models/votre_modele.pkl'
with open("modelerf2.pkl", 'rb') as model_file:
    clf = pickle.load(model_file)

# Charger le modèle préalablement entraîné
#with open("modeleclf.pkl", 'rb') as model_file:
#    clf = pickle.load(model_file)

# Charger vos données (vous pouvez adapter cela si vous souhaitez charger vos données depuis un fichier CSV)
data = pd.read_csv('train.csv')

# fonction de prediction
def predict_credit(feature):
    pred = clf.predict(feature)
    return pred[0]

def main():
    # Page d'accueil de l'application
    st.title("Application de Prédiction de personne en défaut de paiement")

    # Formulaire pour saisir les caractéristiques
    st.sidebar.title("🛠️⚙️Paramètres")
    st.sidebar.header("Saisissez les caractéristiques :")

    Loan_Amount = st.sidebar.number_input('Montant du prêt appliqué', min_value=int(data['Loan Amount'].min()))

    Term_unique = data['Term'].unique()
    Term_unique.sort()
    Term = st.sidebar.number_input(f'Durée du prêt (en mois). Choisir entre {Term_unique}')

    Interest_Rate = st.sidebar.number_input("Le taux d'intérêt")

    Grade_unique = data['Grade'].unique()
    Grade_unique.sort()
    Grade = st.sidebar.selectbox('Note de la banque', Grade_unique)

    Inquires_unique = data['Inquires - six months'].unique()
    Inquires = st.sidebar.slider('nombre total de demandes au cours des 6 derniers mois',
                                 min_value=int(data['Inquires - six months'].min()),
                                 max_value=int(data['Inquires - six months'].max()), value=1)

    Total_Current_Balance = st.sidebar.number_input('Solde courant total de tous les comptes', min_value=int(data['Total Current Balance'].min()))

    Verification_Status_unique =data['Verification Status'].unique()
    Verification_Status = st.sidebar.selectbox('Verification du statut', Verification_Status_unique)

    Application_Type_unique = data['Application Type'].unique()
    Application_Type = st.sidebar.selectbox('Le représentant est un particulier ou une société conjointe', Application_Type_unique)

    Total_Accounts = st.sidebar.number_input('Total de compte')

    Employment_Duration_unique = data['Employment Duration'].unique()
    Employment_Duration = st.sidebar.selectbox("Durée d'emploi", Employment_Duration_unique)

    Public_Record = st.sidebar.number_input("Nombre de documents publics désobligeants")

    Initial_List_Status_unique = data['Initial List Status']. unique()
    Initial_List_Status = st.sidebar.selectbox("Statut de liste unique du prêt - - W (En attente), F (Transféré)", Initial_List_Status_unique)

    Delinquency = st.sidebar.number_input("Nombre de défauts de paiement de plus de 30 jours au cours des 2 dernières années")

    # Ajouter un bouton de validation
    if st.sidebar.button("Prédire"):

        # Préparer les données pour la prédiction
        # Encoder les valeurs avec LabelEncoder

        le_app = LabelEncoder()
        le_app.fit(Application_Type_unique)
        Application_Type_encode = le_app.transform([Application_Type])[0].astype(float)

        le_veri_sta = LabelEncoder()
        le_veri_sta.fit(Verification_Status_unique)
        Verification_Status_encode = le_veri_sta.transform([Verification_Status])[0].astype(float)

        le_grade = LabelEncoder()
        le_grade.fit(Grade_unique)
        Grade_encode = le_grade.transform([Grade])[0].astype(float)

        le_empl = LabelEncoder()
        le_empl.fit(Employment_Duration_unique)
        Employment_Duration_encode = le_empl.transform([Employment_Duration])[0].astype(float)

        le_in_list = LabelEncoder()
        le_in_list.fit(Initial_List_Status_unique)
        Initial_List_Status_encode = le_in_list.transform([Initial_List_Status])[0].astype(float)

        features = [Loan_Amount, Term, Interest_Rate, Grade_encode, Employment_Duration_encode, Inquires, Public_Record, Total_Current_Balance,
                    Application_Type_encode, Initial_List_Status_encode, Verification_Status_encode, Delinquency, Total_Accounts]

        # Préparer les données pour la prédiction
        input_data = [features]

        # Faire la prédiction
        prediction = predict_credit(input_data)

        # Afficher la prédiction
        st.header("Résultat de la prédiction :")
        st.write(f"Cette personne {'est en' if prediction else 'nest pas en'} défaut de paiement.")

if __name__ == '__main__':
    main()
