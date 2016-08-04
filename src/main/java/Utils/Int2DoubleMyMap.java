package Utils;

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

/**
 * Created by 王羚宇 on 2016/8/4.
 */
@SuppressWarnings("unused")
public class Int2DoubleMyMap extends Int2DoubleOpenHashMap {

    public int[] getKeys() {
        return key;
    }

    public double[] getValues() {
        return value;
    }
}
