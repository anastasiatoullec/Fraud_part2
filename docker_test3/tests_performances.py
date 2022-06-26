import os
import requests
import pytest

# définition de l'adresse de l'API
api_address = '127.0.0.1'
# port de l'API
api_port = 8000

@pytest.mark.parametrize("endpoint,model_name", [('log','logistic regression'), ('svm', 'support vector machines'),('tree', 'decision tree classifier'), ('knc', 'K Nearest Neighbors Classifier')])
def test_performances(endpoint,model_name):
    url='http://{address}:{port}/{endpoint}'.format(address=api_address, port=api_port, endpoint=endpoint)
    
    r = requests.get(url)

    output = '''
    ========================================================
        Performances test for {model_name}
    ========================================================

    request done at "/{endpoint}"

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
    output = output.format(status_code=status_code, test_status=test_status, model_name=model_name, endpoint=endpoint)
    print(output)
    
    #ecriture des résultats tests dans le fichier api_test.log
    with open('../api_test.log', 'a') as file:
        file.write(output)

    #assertion pytest
    assert(status_code == 200)

