from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
ml = discovery.build('ml','v1')

def predict():
    item = {"x": [-0.23477546, -0.4932691 ,  1.23672756, -2.33879318, -1.17673345, 0.88573295, -1.96098116, -2.36341211, -2.69477418,  0.36021476, 1.61549548,  0.44775205,  0.60569248,  0.16959132, -0.07365524, -0.16345899,  0.56242311, -0.57703178, -1.63563411,  0.36467924, -1.4953583 , -0.08306639,  0.07461232, -0.34732949,  0.54189984, -0.43329449,  0.08929321, -0.10854652]}
    name = 'projects/{}/models/{}'.format('presentation-212517', 'deployed_classifier')
    name += '/versions/{}'.format('version2')
    response = ml.projects().predict(name=name, body={"instances": item}).execute()
    return response['predictions']
