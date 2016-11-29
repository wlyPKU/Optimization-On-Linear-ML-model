package Utils;

import math.SparseVector;

/**
 * Created by leleyu on 2016/6/30.
 */
public class LabeledData {

  public SparseVector data;
  public double label;
  private double residual;
  public LabeledData(SparseVector data, double label) {
    this.data = data;
    this.label = label;
    this.residual = label;
  }
}
