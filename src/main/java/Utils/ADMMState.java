package Utils;

import math.DenseVector;

/**
 * Created by 王羚宇 on 2016/7/25.
 */
public class ADMMState {
    //B-X C-Z L-U
    public DenseVector x;
    public DenseVector z;
    public DenseVector u;
    public int featureNum;
    public ADMMState(int dimension){
        x = new DenseVector(dimension);
        z = new DenseVector(dimension);
        u = new DenseVector(dimension);
        featureNum = dimension;
    }
}
