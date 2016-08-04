package math;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

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
        if(map.size() < other.map.size()){
            ObjectIterator<Int2DoubleMap.Entry> iter =  map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double value = entry.getDoubleValue();
                if(other.map.containsKey(idx)){
                    double otherValue = other.map.get(idx);
                    result += value * otherValue;
                }
            }
        }else{
            ObjectIterator<Int2DoubleMap.Entry> iter =  other.map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double otherValue = entry.getDoubleValue();
                if(map.containsKey(idx)){
                    double value = map.get(idx);
                    result += value * otherValue;
                }
            }
        }
        return result;
    }
}
