import os
import requests
import fraud as fr

# définition de l'adresse de l'API
api_address = '127.0.0.1'
# port de l'API
api_port = 8000

X_train, y_train, X_test_disp, y_test= fr.preparing_data()


# requête
data= { "user_id": X_test_disp.user_id[0],
        "signup_day": X_test_disp.signup_day[0],
        "signup_month": X_test_disp.signup_month[0],
        "signup_year": X_test_disp.signup_year[0],
        "purchase_day": X_test_disp.purchase_day[0],
        "purchase_month": X_test_disp.purchase_month[0],
        "purchase_year": X_test_disp.purchase_year[0],
        "purchase_value": X_test_disp.purchase_value[0],
        "age": X_test_disp.age[0],
        "source_Ads": int(X_test_disp.source_Ads[0]),
        "source_Direct": int(X_test_disp.source_Direct[0]),
        "source_SEO": int(X_test_disp.source_SEO[0]),
        "browser_Chrome": int(X_test_disp.browser_Chrome[0]),
        "browser_FireFox": int(X_test_disp.browser_FireFox[0]),
        "browser_IE": int(X_test_disp.browser_IE[0]),
        "browser_Opera": int(X_test_disp.browser_Opera[0]),
        "browser_Safari": int(X_test_disp.browser_Safari[0]),
        "sex_F": int(X_test_disp.sex_F[0]),
        "sex_M": int(X_test_disp.sex_M[0])
}
    
url='http://127.0.0.1:8000/predict3'

r = requests.post(url, json=data)

output = '''
========================================================
    Prediction fraud test for logistic regression
========================================================

request done at "/prediction"

expected result = 200
actual restult = {status_code}

==>  {test_status}

'''
# statut de la requête
status_code = r.status_code

# affichage des résultats
if status_code == 200:
    test_status = 'SUCCESS'
else:
    test_status = 'FAILURE'
print(output.format(status_code=status_code, test_status=test_status))
