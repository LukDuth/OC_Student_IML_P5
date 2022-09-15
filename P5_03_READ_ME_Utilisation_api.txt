--Pré requis au fonctionnement de l'API--

Pour fonctionner l'API a besoin des deux fichiers V04test_feat_use et V04test_moc_logreg_use, dont elle charge le contenu grâce a des fonctions du module python pickle.

Les chemin d'accès à ces deux fichiers est écrit explicitement dans le scirpt python de P5_03_api_streamlit_luke.py, sous forme de variables de type chaînes de caractères. 
Si vous ne voulez pas vous embêtez à recréer des répertoires correspondant à ces chemin (de types ../Cache_fichiers/Test_models/) vous pouvez modifiez ces variables en leur assignant le répertoire courant : ".".

Cette API est basée sur de l'extraction de features par la méthode de words embedding dite USE. Elle nécessite le chargement d'un modèle pré-entraîné, ce qui est fait via une ligne python du script :
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
Cette ligne est également présente dans le notebook de test de modèles. Le modèle correspondant sera chargé normalement dans un répertoire de stockage temporaire de votre ordinateur, de type :C:/Users/luked/AppData/Local/Temp/tfhub_modules/063d866c06683311b44b4992fd46003be952409c/
Attention, le stockage est temporaire, le contenu de ce dossier finit par s'effacer seul. Pas de panique, si c'est le cas, il suffit de supprimer le répertoire /063d866c06683311b44b4992fd46003be952409c/, puis de relancer la ligne embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") (et d'attendre une dizaine de minutes que le chargement se refasse).

--Lancement de l'API--
Depuis un terminal ouvert dans le répertoire contenant le script python, et de taper les commandes suivantes :
"streamlit run P5_03_api_streamlit_luke.py"
(sous couvert d'avoir bien installé streamlit sur son environnement virtuel)
