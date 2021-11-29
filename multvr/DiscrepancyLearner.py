import GeometryUtils as GeoUtils

class DiscrepancyLearner:
    def __init__(self):


    def _determineInitialSamplePoints(self, initial_set: GeoUtils.SSSlice):
        initial_set.wrap("Axis-Aligned Ellipsoid")
