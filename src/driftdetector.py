from river.drift import ADWIN, PageHinkley

class DriftDetector:
    def __init__(self, method='ADWIN', **kwargs):
        if method == 'ADWIN':
            self.detector = ADWIN(**kwargs)
        elif method == 'PageHinkley':
            self.detector = PageHinkley(**kwargs)
        else:
            raise ValueError(f"Unknown drift detector {method}")
        self.method = method

    def update(self, value):
        self.detector.update(value)

        if self.method == 'ADWIN':
            drift = self.detector.drift_detected
            magnitude = self.detector.width 
            return drift, False, magnitude

        elif self.method == 'PageHinkley':
            drift = self.detector.drift_detected
            # Approximate drift magnitude from the absolute cumulative sum
            magnitude = abs(self.detector.sum) if drift else 0
            return drift, False, magnitude
