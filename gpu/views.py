import GPUtil
from django.http import JsonResponse
import random

class GPUStatus:
  def __init__(self, id, gpu_util, free_memory, used_memory, total_memory):
    self.id = id
    self.gpu_util = gpu_util
    self.free_memory = free_memory
    self.used_memory = used_memory
    self.total_memory = total_memory

  def to_dict(self):
    return {
      "id": self.id,
      "gpu_util": self.gpu_util,
      "free_memory": self.free_memory,
      "used_memory": self.used_memory,
      "total_memory": self.total_memory,
    }

def get_gpu(request):
  min_memory_gpu_avail = 40000  
  gpu_list = []
  gpus = GPUtil.getGPUs()
  gpu_avail = 0
  for gpu in gpus:
    if (gpu.memoryFree > min_memory_gpu_avail):
      gpu_avail += 1
      gpu_object = GPUStatus(
          id=gpu.id,
          gpu_util=gpu.load,
          free_memory=gpu.memoryFree,
          used_memory=gpu.memoryUsed,
          total_memory=gpu.memoryTotal
      )
      gpu_list.append(gpu_object.to_dict())  # Convert to dictionary
    print('gpu list: ', gpu_list)

  # Testing purpose

  data = {
    "total_gpu_available": gpu_avail,
    "gpu": gpu_list
  }
  return JsonResponse(data)