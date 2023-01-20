



class SamplerWrapper():
    def __init__(self, name, config):
        self.sampler = self._set_sampler(name, config)

    def _set_sampler(self, name, config):
        if name == "nthstep":
            return NthStepSampler(**config)
        elif name == "range":
            return RangeSampler(**config)
        elif name == "cluster":
            return ClusterSampler(**config)
        elif name == "voxelgrid":
            return VoxelGridSampler(**config)
        elif name == "maxnum":
            return MaxNumStepSampler(**config)

    def sample(self, input):
        return self.sampler.sample(input)

class NthStepSampler():
    def __init__(self, step_size=[5, 5]):
        self.step_size = step_size
    def sample(self, input):
        row_sample_indices = list(range(0, input.shape[1], self.step_size[0]))
        column_sample_indices = list(range(0, input.shape[2], self.step_size[1]))

        column_sample_indices_final = []
        for i in range(len(column_sample_indices)):
            # for j in range(len(coz)
            column_sample_indices_final = column_sample_indices_final + ([column_sample_indices[i]] * len(row_sample_indices))
        row_sample_indices_final = row_sample_indices * len(column_sample_indices)

        return input[:, row_sample_indices_final, column_sample_indices_final]

class MaxNumStepSampler():
    def __init__(self, max_steps=[150, 150]):
        self.max_steps = max_steps
    def sample(self, input):
        row_sample_indices = list(range(0, input.shape[1], int(round(input.shape[1]/self.max_steps[0]))))
        column_sample_indices = list(range(0, input.shape[2], int(round(input.shape[2]/self.max_steps[1]))))

        column_sample_indices_final = []
        for i in range(len(column_sample_indices)):
            # for j in range(len(coz)
            column_sample_indices_final = column_sample_indices_final + ([column_sample_indices[i]] * len(row_sample_indices))
        row_sample_indices_final = row_sample_indices * len(column_sample_indices)

        return input[:, row_sample_indices_final, column_sample_indices_final]

class RangeSampler():
    def __init__(self, ranges=[[0, 150], [0, 150]]):
        self.ranges = ranges
    def sample(self, input):
        return input[:, self.ranges[0][0]:self.ranges[0][1], self.ranges[1][0]:self.ranges[1][1]]

# specify/create detailed version of clustersampler related to used clustering algorithm e.g. KMeansSampler
class ClusterSampler():
    def __init__(self):
        raise NameError("Not implemented yet!")
    def sample(self, input):
        raise NameError("Not implemented yet!")

class VoxelGridSampler():
    def __init__(self):
        raise NameError("Not implemented yet!")
    def sample(self, input):
        raise NameError("Not implemented yet!")