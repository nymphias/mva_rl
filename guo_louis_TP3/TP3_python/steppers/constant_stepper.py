import steppers
class constant_stepper(steppers.stepper):
    def __init__(self, lrate):
        self.lrate = lrate

    def update(self, gt):
        return self.lrate * gt

