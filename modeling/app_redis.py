import redis

class Redis:
    r = None

    @classmethod
    def _initialize(self):
        if self.r is None:
            self.r = redis.Redis(host='localhost', port=6379)
            try:
                if self.r.ping():
                    print("Redis Ready!")
            except redis.ConnectionError:
                print('Failed to connect to redis')
                self.r = None
    
    @classmethod
    def get(self):
        return self.r
    
