from django.shortcuts import render
import os
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view

@api_view(['POST'])
def delete_model(req):
  try:
    print('hei')
    workspace_type = req.data['workspace_type']
    print(workspace_type)
    if (workspace_type == 'object_segmentation'):
      model_name = req.data['model_name']
      username = req.data['username']
      workspace = req.data['workspace']

      try:
        base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_file = os.path.join(base_directory, 'ml_model', 'models', 'weights', f'{model_name}_{username}_{workspace}.pth')
        print(weights_file)
        os.remove(weights_file)
        return Response({'message': "deleted successfully"},status=status.HTTP_204_NO_CONTENT)
      except:
        return Response({'message': "data not found, pth file already deleted"}, status=status.HTTP_404_NOT_FOUND)
    else: 
      return Response({'message': "workspace type should be object segmentation"}, status=status.HTTP_400_BAD_REQUEST)
  except:
    return Response({'message': "input error"}, status=status.HTTP_400_BAD_REQUEST)