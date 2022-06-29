import os
import requests

data_fraud= {  
    "user_id": 1359,
    "signup_day": 1,
    "signup_month": 1,
    "signup_year": 2015,
    "purchase_day": 1,
    "purchase_month": 1,
    "purchase_year": 2015,
    "purchase_value": 15,
    "source": "SEO",
    "browser": "Opera",
    "sex": "M",
    "age": 53
    }

# définition de l'adresse de l'API
api_address = 'api_container'
# port de l'API
api_port = 8000


def test_fraud_log():
    
    url1='http://{address}:{port}/predict1'.format(address=api_address, port=api_port)

    

    r = requests.post(url1, json=data_fraud)

    output = '''
    =============================================================
        Fraud prediction test status code for logistic regression
    =============================================================

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

    output = output.format(status_code=status_code, test_status=test_status)
    print(output)

    with open('../api_test.log', 'a') as file:
        file.write(output)

    output = '''
    ========================================================
        Fraud prediction test target for logistic regression
    ========================================================

    request done at "/predict1"

    expected result = [1]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [1]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    
    output = output.format(value=value, test_status=test_status)
    print(output)

    with open('../api_test.log', 'a') as file:
        file.write(output)
    
    assert(status_code == 200)
    assert(value == [1])

def test_fraud_svm():
    url2='http://{address}:{port}/predict2'.format(address=api_address, port=api_port)

    r = requests.post(url2, json=data_fraud)

    output = '''
    =================================================================
        Fraud prediction test status code for support vector machines
    =================================================================

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

    output = output.format(status_code=status_code, test_status=test_status)
    print(output)


    with open('../api_test.log', 'a') as file:
        file.write(output)

    output = '''
    =============================================================
        Fraud prediction test target for support vector machines
    =============================================================

    request done at "/predict2"

    expected result = [1]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [1]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    output = output.format(value=value, test_status=test_status)
    print(output)
    with open('../api_test.log', 'a') as file:
        file.write(output)

    assert(status_code == 200)
    assert(value == [1])

def test_fraud_tree():

    url3='http://{address}:{port}/predict3'.format(address=api_address, port=api_port)

    r = requests.post(url3, json=data_fraud)

    output = '''
    =============================================================
        Fraud prediction test for decision tree classifier
    =============================================================

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

    output = output.format(status_code=status_code, test_status=test_status)
    print(output)
    with open('../api_test.log', 'a') as file:
        file.write(output)

    output = '''
    ========================================================
        Fraud prediction test for decision tree classifier
    ========================================================

    request done at "/predict3"

    expected result = [1]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [1]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'

    output = output.format(value=value, test_status=test_status)
    print(output)
    with open('../api_test.log', 'a') as file:
        file.write(output)

    assert(status_code == 200)
    assert(value == [1])

def test_fraud_knn():

    url4='http://{address}:{port}/predict4'.format(address=api_address, port=api_port)

    r = requests.post(url4, json=data_fraud)

    output = '''
    ==================================================================
        Fraud prediction test for K Nearest Neighbors Classifier
    ==================================================================

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
    output = output.format(status_code=status_code, test_status=test_status)
    print(output)
    with open('../api_test.log', 'a') as file:
        file.write(output)

    output = '''
    ========================================================
        Fraud prediction test for K Nearest Neighbors Classifier
    ========================================================

    request done at "/predict4"

    expected result = [1]
    actual result = {value}

    ==>  {test_status}

    '''
    # valeur target
    data = r.json()
    value = data['Predicted transaction(1 - fraud, 0 - not fraud)']

    # affichage des résultats
    if value == [1]:
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'
    output = output.format(value=value, test_status=test_status)
    print(output)
    with open('../api_test.log', 'a') as file:
        file.write(output)
    
    assert(status_code == 200)
    assert(value == [1])
    