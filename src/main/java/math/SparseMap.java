package math;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class SparseMap {
    public Int2DoubleOpenHashMap map;
    public SparseMap(){
        map = new Int2DoubleOpenHashMap();
    }
    public void add(int i, double v){
        map.put(i, v);
    }
    public double multiply(SparseMap other){
        double result = 0;
        int ite1 = 0, ite2 = 0;
        int []index1 = map.keySet().toIntArray();
        int []index2 = other.map.keySet().toIntArray();
        while(ite1 < index1.length && ite2 < index2.length){
            if(index1[ite1] < index2[ite2]){
                ite1++;
            }else if(index1[ite1] > index2[ite2]){
                ite2++;
            }else if(index1[ite1] == index2[ite2]){
                result += map.get(index1[ite1]) * other.map.get(index2[ite2]);
                ite1++;
                ite2++;
            }
        }
        return result;
    }
}
