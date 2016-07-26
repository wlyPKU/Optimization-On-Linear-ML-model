package Utils;

import math.DenseVector;

/**
 * Created by 王羚宇 on 2016/7/25.
 */
public class ADMMState {
    public DenseVector B;
    public DenseVector C;
    public DenseVector L;
    public ADMMState(int dimension){
        B = new DenseVector(dimension);
        C = new DenseVector(dimension);
        L = new DenseVector(dimension);

    }
}
