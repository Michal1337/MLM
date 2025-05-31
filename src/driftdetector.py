from river.drift import ADWIN, PageHinkley

class DriftDetector:
    def __init__(self, method='ADWIN'):
        if method == 'ADWIN':
            self.detector = ADWIN(clock=1)
        elif method == 'PageHinkley':
            self.detector = PageHinkley()
        else:
            raise ValueError(f"Unknown drift detector {method}")
        self.method = method

    def update(self, value):
        self.detector.update(value)
        return self.detector.drift_detected
