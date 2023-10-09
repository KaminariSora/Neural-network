df = pd.read_csv('DATASET\y_wineQT.csv')# อ่านไฟล์
# features = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
#                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol' ]].values.tolist()# ดึงข้อมูล Column เป็น Array
# labels = df[['quality']].values.tolist()# Result = quality
# classifier = tree.DecisionTreeClassifier()# train with DecisionTreeClassifier
# classifier = classifier.fit(features, labels)# เทียบข้อมูลแต่ละ Arrays 
# joblib.dump(classifier, 'model.pkl')# export model