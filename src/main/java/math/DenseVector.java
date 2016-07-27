package math;

/**
 * Created by leleyu on 2016/6/30.
 */
public class DenseVector {
  public int dim;
  public double[] values;

  public DenseVector(int dim) {
    this.dim = dim;
    this.values = new double[dim];
  }

  public double dot(SparseVector other) {
    int[] indices = other.indices;
    double ret = 0.0;
    if (other.values != null) {
      for (int i = 0; i < indices.length; i ++) {
        ret += values[indices[i]] * other.values[i];
      }
    } else {
      for (int i = 0; i < indices.length; i ++) {
        ret += values[indices[i]];
      }
    }
    return ret;
  }

  public void plusBy(SparseVector other, double x) {
    int[] indices = other.indices;
    for (int i = 0; i < indices.length; i ++) {
      values[indices[i]] += x;
    }
  }
  public void set(int idx, double value){

  }
  public void positiveValueOrZero(SparseVector other) {
    int[] indices = other.indices;
    for(int i = 0; i < indices.length; i ++){
      if(values[indices[i]] < 0){
        values[indices[i]] = 0;
      }
    }
  }
  public void plusGradient(SparseVector other, double scala){
    for(int i = 0; i <  other.indices.length; i++){
      int idx = other.indices[i];
      if(other.values != null){
        values[idx] += scala * other.values[i];
      }else{
        values[idx] += scala;
      }
    }
  }
}
