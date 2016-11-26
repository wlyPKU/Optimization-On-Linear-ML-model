package test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/11/26.
 */
public class CollectionsDemo {
    public static void main(String args[]) {
        // create array list object
        List arrlist = new ArrayList();

        // populate the list
        arrlist.add("A");
        arrlist.add("B");
        arrlist.add("C");

        System.out.println("Initial collection: "+arrlist);

        // shuffle the list
        Collections.shuffle(arrlist);
        for(int i = 0; i < arrlist.size(); i++){
            System.out.print(arrlist.get(i));
        }
        System.out.println("Final collection after shuffle: "+arrlist);
    }
}
