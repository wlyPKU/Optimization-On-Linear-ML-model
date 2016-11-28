package Utils;

import math.DenseVector;

import java.util.Arrays;

/**
 * Created by 王羚宇 on 2016/7/25.
 */
public class ADMMFeatureState {
    //B-X C-Z L-U
    public DenseVector x;
    public DenseVector z;
    public DenseVector u;
    public int beginOffset;
    public double AX[];
    public double globalAX[];
    public int featureDimension;
    public int sampleDimension;
    public ADMMFeatureState(int featureDimension, int sampleDimension, int beginOffset){
        x = new DenseVector(featureDimension);
        z = new DenseVector(sampleDimension);
        u = new DenseVector(sampleDimension);
        this.featureDimension = featureDimension;
        this.beginOffset = beginOffset;
        this.sampleDimension = sampleDimension;
        AX = new double[sampleDimension];
        globalAX = new double[sampleDimension];
        Arrays.fill(AX, 0);
        Arrays.fill(globalAX, 0);
    }
}
