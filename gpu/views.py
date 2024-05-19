import GPUtil
from pydantic import BaseModel
from rest_framework.views import APIView
from rest_framework.response import Response

GPUtil.showUtilization()

class GPUStatus(BaseModel):
  id: int
  gpu_util: float
  free_memory: float
  used_memory: float
  total_memory: float

class GPUChecker(APIView):
  def get(self, request):
    gpu_list = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
      gpu_object = GPUStatus(
        id = gpu.id,
        gpu_util = gpu.load,
        free_memory = gpu.memoryFree,
        used_memory = gpu.memoryUsed,
        total_memory = gpu.memoryTotal
      )
      gpu_list.append(gpu_object)
      print('gpu list: ', gpu_list)
    # return gpu_list
    return Response(gpu_list)