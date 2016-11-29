package math;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

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
        ObjectIterator<Int2DoubleMap.Entry> iter =  map.int2DoubleEntrySet().iterator();
        ObjectIterator<Int2DoubleMap.Entry> anotherIter =  other.map.int2DoubleEntrySet().iterator();
        Int2DoubleMap.Entry entry = null;
        Int2DoubleMap.Entry anotherEntry = null;
        if(iter.hasNext()) {
            entry = iter.next();
        }
        if(anotherIter.hasNext()) {
            anotherEntry = anotherIter.next();
        }
        while (entry != null && (anotherEntry != null)) {
            int idx = entry.getIntKey();
            int anotherIdx = anotherEntry.getIntKey();

            if (idx < anotherIdx) {
                entry = iter.hasNext()?iter.next():null;
            }else if(idx > anotherIdx){
                anotherEntry = anotherIter.hasNext()?anotherIter.next():null;
            }else{
                double value = entry.getDoubleValue();
                double anotherValue = anotherEntry.getDoubleValue();
                result += value * anotherValue;
                entry = iter.hasNext()?iter.next():null;
                anotherEntry = anotherIter.hasNext()?anotherIter.next():null;
            }
        }
        return result;
    }
}
