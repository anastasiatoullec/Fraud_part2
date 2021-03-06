import os
import requests
import time

# définition de l'adresse de l'API
api_address = 'api_container'
# port de l'API
api_port = 8000

data_not_fraud= {  
  "user_id": 22058,
  "signup_day": 24,
  "signup_month": 2,
  "signup_year": 2015,
  "purchase_day": 18,
  "purchase_month": 4,
  "purchase_year": 2015,
  "purchase_value": 34,
  "source": "SEO",
  "browser": "Chrome",
  "sex": "M",
  "age": 39
}

time.sleep(5)

def test_not_fraud_log():

    url1='http://{address}:{port}/predict1'.format(address=api_address, port=api_port)

    r = requests.post(url1, json=data_not_fraud)

    output = '''
    ==================================================================
        Not fraud prediction test stattus code for logistic regression
    ==================================================================

    request done at "/predict1"

    expected result = 200
    actual result = {status_code}

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

    output = '''
    ============================================================
        Not fraud prediction test target for logistic regression
    ============================================================

    request done at "/predict1"

    expected result = [0]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [0]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    output = output.format(value=value, test_status=test_status)
    print(output)

    #ecriture des résultats tests dans le fichier api_test.log
    with open('../api_test.log', 'a') as file:
        file.write(output)

    #assertion pytest
    assert(status_code == 200)
    assert(value == [0])

def test_not_fraud_svm():
    url2='http://{address}:{port}/predict2'.format(address=api_address, port=api_port)

    r = requests.post(url2, json=data_not_fraud)

    output = '''
    =====================================================================
        Not fraud prediction test status code for support vector machines
    =====================================================================

    request done at "/predict2"

    expected result = 200
    actual result = {status_code}

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


    output = '''
    =================================================================
        Not fraud prediction test target for support vector machines
    =================================================================

    request done at "/predict2"

    expected result = [0]
    actual result = {value}

    ==>  {test_status}

    '''

    print(r.json())
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [0]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    
    output = output.format(value=value, test_status=test_status)
    print(output)

    #ecriture des résultats tests dans le fichier api_test.log
    with open('../api_test.log', 'a') as file:
        file.write(output)

    #assertion pytest
    assert(status_code == 200)
    assert(value == [0])

def test_not_fraud_tree():
    url3='http://{address}:{port}/predict3'.format(address=api_address, port=api_port)

    r = requests.post(url3, json=data_not_fraud)

    output = '''
    ======================================================================
        Not fraud prediction test status code for decision tree classifier
    ======================================================================

    request done at "/predict3"

    expected result = 200
    actual result = {status_code}

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

    output = '''
    =================================================================
        Not fraud prediction test target for decision tree classifier
    =================================================================

    request done at "/predict3"

    expected result = [0]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [0]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    output = output.format(value=value, test_status=test_status)
    print(output)

    #ecriture des résultats tests dans le fichier api_test.log
    with open('../api_test.log', 'a') as file:
        file.write(output)

    #assertion pytest
    assert(status_code == 200)
    assert(value == [0])

def test_not_fraud_knn():
    url4='http://{address}:{port}/predict4'.format(address=api_address, port=api_port)

    r = requests.post(url4, json=data_not_fraud)

    output = '''
    ===========================================================================
        Not fraud prediction test status cde for K Nearest Neighbors Classifier
    ===========================================================================

    request done at "/predict4"

    expected result = 200
    actual result = {status_code}

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

    output = '''
    ========================================================================
        Not fraud prediction test target for K Nearest Neighbors Classifier
    ========================================================================

    request done at "/predict4"

    expected result = [0]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [0]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    output = output.format(value=value, test_status=test_status)
    print(output)

      #ecriture des résultats tests dans le fichier api_test.log
    with open('../api_test.log', 'a') as file:
        file.write(output)

    #assertion pytest
    assert(status_code == 200)
    assert(value == [0])