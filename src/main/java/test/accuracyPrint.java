package test;

/**
 * Created by 王羚宇 on 2017/1/9.
 */
public class accuracyPrint {
    public static void main(String args[]) {
        for(double i = 1; i > 0.0000000001; i *=0.1){
                double exp2 = Math.exp(0.5 * i);
                System.out.println(i + " " + exp2);
        }
    }
}
