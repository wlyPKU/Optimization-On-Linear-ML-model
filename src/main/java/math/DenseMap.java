package math;

import java.util.*;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class DenseMap {
    public List<Integer> index;
    public List<Double> value;
    public DenseMap(){
        index = new ArrayList<Integer>();
        value = new ArrayList<Double>();
    }
    public void add(int i, double v){
        index.add(i);
        value.add(v);
    }
    public double multiply(DenseMap other){
        double result = 0;
        int ite1 = 0, ite2 = 0;
        while(ite1 < index.size() && ite2 < other.index.size()){
            if(index.get(ite1) < other.index.get(ite2)){
                ite1++;
            }else if(index.get(ite1) > other.index.get(ite2)){
                ite2++;
            }else if(index.get(ite1).equals(other.index.get(ite2)) ){
                result += value.get(ite1) * other.value.get(ite2);
                ite1++;
                ite2++;
            }
        }
        return result;
    }
}
