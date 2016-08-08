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

  public DenseVector(DenseVector a){
    this.dim = a.dim;
    this.values = new double[dim];
    System.arraycopy(a.values, 0, values, 0, dim);
  }
  public double dot(SparseVector other) {
    int[] indices = other.indices;
    double ret = 0.0;
    if (other.values != null) {
      for (int i = 0; i < indices.length; i ++) {
        ret += values[indices[i]] * other.values[i];
      }
    } else {
      for (int i: indices) {
        ret += values[i];
      }
    }
    return ret;
  }

  public void plusBy(SparseVector other, double x) {
    int[] indices = other.indices;
    for (int i: indices) {
      values[i] += x;
    }
  }

  public void allPlusBy(double x){
    for(int i = 0; i < values.length; i++){
      values[i] += x;
    }
  }
  public void positiveValueOrZero() {
    for (int i = 0; i < values.length; i++) {
      values[i] = Math.max(0, values[i]);
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
