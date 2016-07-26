package Utils;

import math.SparseVector;

/**
 * Created by leleyu on 2016/6/30.
 */
public class MinHash {

  public int numPerm;
  public int dim;
  public int b;
  public int[][] permutations;

  public int[] integersTempArray;

  public MinHash(int K, int dim, int b) {
    this.numPerm = K;
    this.dim = dim;
    this.b = b;
    permutations = new int[numPerm][];
    for (int i = 0; i < numPerm; i ++) {
      permutations[i] = Utils.generateRandomPermutation(dim);
    }
    integersTempArray = new int[K];
  }

  private int[] generateIntegers(SparseVector data) {
    for (int i = 0; i < numPerm; i ++) {
      int[] permutation = permutations[i];
      int minIdx = Integer.MAX_VALUE;
      int[] indices = data.indices;
      for (int j = 0; j < indices.length; j ++) {
        if (permutation[indices[j]] < minIdx) {
          minIdx = permutation[indices[j]];
        }
      }
      integersTempArray[i] = minIdx;
    }

    return integersTempArray;
  }

  private static int getLastBits(int num, int b) {
    int a = 1;
    for (int i = 1; i < b; i ++)
      a += 1 << i;
    return num & a;
  }

  private int[] generateBinaryBits(int[] ints) {
    int segNum = (int) Math.pow(2, b);
    int[] bitIdx = new int[ints.length];
    for (int i = 0; i < ints.length; i ++) {
      int idx = getLastBits(ints[i], b);
      bitIdx[i] = i * segNum + idx;
    }
    return bitIdx;
  }

  public static void printBits(int num) {
    byte[] bits = new byte[32];

    for (int i = 0; i < 32; i ++) {
      if ((num & (1 << (31 - i))) > 0) {
        bits[i] = 1;
      } else
        bits[i] = 0;
    }

    StringBuffer sb = new StringBuffer(32);
    for (int i = 0; i < 32; i ++) {
      sb.append(bits[i]);
    }

    System.out.println(sb.toString());
  }

  public int[] generateMinHashBits(SparseVector data) {
    int[] ints = generateIntegers(data);
    return generateBinaryBits(ints);
  }

  public int getHashedDim() {
    return (int) Math.pow(2, b) * numPerm;
  }

  public static void main(String[] argv) {
    printBits(7);
    printBits(getLastBits(7, 2));
    printBits(10);
    printBits(getLastBits(10, 3));
  }

}
