class PID():
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01, max_u = 50):
        self.Kp = Kp # 비례 gain
        self.Ki = Ki # 적분 gain
        self.Kd = Kd # 미분 gain
        self.dt = dt # 시간 간격
        self.max_u = max_u # 최대 제어 신호 크기. 시뮬레이터에서 최대 조향각

        self.i_err = 0 # error의 적분값
        self.prev_err = 0 # 이전의 error 값

    def do(self, r, y, de):
        err = r - y
        up = self.Kp * err 
        ui = self.Ki * self.i_err
        ud = self.Kd * de

        u = up + ui + ud

        u = max(min(u, self.max_u), -self.max_u)
        self.i_err += err * self.dt 
        
        return u 