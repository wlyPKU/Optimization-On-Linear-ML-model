package test;

import java.util.*;

/**
 * Created by 王羚宇 on 2016/11/26.
 */
public class randomTest {
    public static int randomNumber(int numberBits){
        Random random = new Random();
        int number = 0;
        int ite = 0;
        while(ite < numberBits){
            int r = random.nextInt(2);
            number = number * 2 + r;
            ite++;
        }
        return number;
    }

    public static int random(int a, int b){
        int interval = b - a + 1;
        int standardInterval = 0;
        while(Math.pow(2.0, standardInterval) < interval){
            standardInterval ++;
        }
        int candidate = randomNumber(standardInterval);
        while (candidate >= interval){
            candidate = randomNumber(standardInterval);
        }
        return candidate + a;
    }
    public static int randomBase0() {
        Random r = new Random();
        return r.nextInt(2);
    }
    public static int randomBaseS() {
        String s = new String(new StringBuffer().append(getBoolean()).append(getBoolean()));

        if("00".equals(s)){
            return 0;
        }else if("01".equals(s)){
            return 1;
        }else if("10".equals(s)){
            return 2;
        }else{
            return 3;
        }
    }
    //获取随机数二进制字符串
    public static String getBoolean(){
        return Integer.toString(randomBase0());
    }
    public static void main(String args[]) {
        int from = Integer.parseInt(args[0]);
        int to = Integer.parseInt(args[1]);
        int result[] = new int[to - from + 1];
        Arrays.fill(result, 0);
        for(int i = 0; i < 100000; i++){
            result[random(from, to)] += 1;
            //result[randomBaseS()] += 1;
        }
        for(int i = 0; i < to - from + 1; i++){
            System.out.println((from + i) + " :" + result[i]);
        }
    }
}
