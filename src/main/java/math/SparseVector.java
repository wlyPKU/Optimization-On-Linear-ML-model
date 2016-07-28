package math;

/**
 * Created by leleyu on 2016/6/30.
 */
public class SparseVector {

  public int[] indices;
  public double[] values;
  public int dim;

  public SparseVector(int dim, int[] indices) {
    this.dim = dim;
    this.indices = indices;
  }
  @SuppressWarnings("unused")
  public SparseVector(int dim, int[] indices, double[] values) {
    this(dim, indices);
    this.values = values;
  }
  @SuppressWarnings("unused")
  public double multiply(SparseVector other){
    assert(dim == other.dim);
    double result = 0;
    int ite1 = 0, ite2 = 0;
    while(ite1 < dim && ite2 < other.dim){
      if(indices[ite1] < other.indices[ite2]){
        ite1++;
      }else if(indices[ite1] > other.indices[ite2]){
        ite2++;
      }else if(indices[ite1] == other.indices[ite2]){
        result += (values == null? 1: values[ite1]) * (other.values == null? 1: other.values[ite2]);
      }
    }
    return result;
  }
}
