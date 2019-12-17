/**
 * 
 */
package mySMO;

import java.util.Vector;
import java.util.*;
import java.io.*;

/**
 * @author haibn
 *
 */

public class SMO {

	int numChanged;
	int examineAll;
	
	int N = 0; //N data point
	int d = -1; //d dimensions of a point
	float C = (float)1;

	float tolerance = (float)0.001;
	float eps = (float)0.001;
	float two_sigma_squared = (float)2;
	
	Vector alpha = new Vector();  //Lagrange multipliers
	float b;  // threshold
	float delta_b=0;   
	Vector w = new Vector(); //weight only for linear kernel
	
	Vector error_cache = new Vector();
	
	public int MATRIX=5000;
	float dense_points[][] = new float[MATRIX][MATRIX];
	
	//tich vo huong duoc tinh san~ va luu trong cac mang nay
	//Vector precomputed_self_dot_product = new Vector();
	float precomputed_dot_product[][] = new float[MATRIX][MATRIX];
	
	final int linear_kernel_flag = 1;
	final int poly_kernel_flag = 2;
	final int rbf_kernel_flag = 3;
	public int kernel_type = rbf_kernel_flag;
	
	public int poly_degree = 3;
	
	//boolean is_linear_kernel = false;
	
	Vector target = new Vector();

	//boolean is_test_only = false;

	// support vectors are within [0..end_support_i)
	int end_support_i = -1;
	//-----------------------------------
	//------------------------------------
	
	int numStep= 0;
	
	int examineExample(int i1)
	{
		float y1, alpha1, E1, R1;
		
		y1 = (float)((Integer)target.get(i1)).intValue();
		alpha1 = ((Float)alpha.get(i1)).floatValue();
		
		if (alpha1 > 0 && alpha1 < C)
			E1 = ((Float)error_cache.get(i1)).floatValue();
		else 
			E1 = learned_func(i1) - y1;
			//System.out.println("learned "+learned_func(i1));
		R1 = y1*E1;
		
		if ((R1 < -tolerance && alpha1 < C) || (R1 > tolerance && alpha1 > 0)) //violate KKT condition
		{       
				int k=0, i2=-1;
			    int k0=0;
				float max=0;
      
				for (k=0; k < end_support_i; k++)
					if (((Float)alpha.get(k)).floatValue() > 0 && ((Float)alpha.get(k)).floatValue() < C) 
					{
						float E2=0, temp=0;
						E2 = ((Float)error_cache.get(k)).floatValue();
						temp = Math.abs(E1 - E2);
						if (temp > max)
						{
							max = temp;
							i2 = k;
      
						}
					}
   
				if (i2 >= 0) 
				{
					if (takeStep (i1, i2)==1)
					{   
						return 1;
					}
				}
				//-----------------------------------------
			    float rand = (float)Math.random();
				k0=(int) (rand * end_support_i);
				i2=0; 
   
				for (k = k0; k < end_support_i + k0; k++) 
				{
    
					i2 = k % end_support_i;
        
					if (((Float)alpha.get(i2)).floatValue() > 0 && ((Float)alpha.get(i2)).floatValue() < C) 
					{
						if (takeStep(i1, i2)==1)
						{
							return 1;
						}
					}
				}
				//---------------------------------------------
				rand = (float)Math.random();
				k0 = (int)(rand * end_support_i);
				i2=0;
				for (k = k0; k < end_support_i + k0; k++) 
				{
					i2 = k % end_support_i;
					if (takeStep(i1, i2)== 1)
					{ 	
       					return 1;
					}
				}
		}
		return 0;
	}
	
	int takeStep(int i1, int i2) 
	{ 
		int y1=0, y2=0, s=0;
		float alpha1=0, alpha2=0; /* old_values of alpha_1, alpha_2 */
		float a1=0, a2=0;       /* new values of alpha_1, alpha_2 */
		float E1=0, E2=0, L=0, H=0, k11=0, k22=0, k12=0, eta=0, Lobj=0, Hobj=0;

		if (i1 == i2) return 0;
		//System.out.println("takeStep");
		alpha1 = ((Float)alpha.get(i1)).floatValue();
		y1 = ((Integer)target.get(i1)).intValue();
		if (alpha1 > 0 && alpha1 < C)
			E1 = ((Float)error_cache.get(i1)).floatValue();
		else 
			E1 = learned_func(i1) - y1;
		
		alpha2 = ((Float)alpha.get(i2)).floatValue();
		y2 = ((Integer)target.get(i2)).intValue();
		if (alpha2 > 0 && alpha2 < C)
			E2 = ((Float)error_cache.get(i2)).floatValue();
		else 
			E2 = learned_func(i2) - y2;
  
		s = y1 * y2;
		// compute L, H
		if (y1 == y2)
		{
			float gamma = alpha1 + alpha2;
			if (gamma > C) 
			{
				L = gamma-C;
				H = C;
			}
			else 
			{
				L = 0;
				H = gamma;
			}
			//if (L == H)
				//System.out.println("gamma "+gamma+"-L "+L+"-H "+H);
		}
		else
		{
			float gamma = alpha1 - alpha2;
			if (gamma > 0) 
			{
				L = 0;
				H = C - gamma;
			}
			else 
			{
				L = -gamma;
				H = C;
			}
			//if (L == H)
				//System.out.println("gamma "+gamma+"-L "+L+"-H "+H);
		}

		if (L == H)	return 0;

		k11 = kernel_func(i1, i1);
		k12 = kernel_func(i1, i2);
		k22 = kernel_func(i2, i2);
		eta = 2 * k12 - k11 - k22;

		if (eta < 0) 
		{
			a2 = alpha2 + (y2 * (E2 - E1) / eta);
			if (a2 < L) a2 = L;
			else if (a2 > H) a2 = H;
		}
		else 
		{
			float c1 = eta/2;
			float c2 = y2 * (E1-E2)- eta * alpha2;
			Lobj = c1 * L * L + c2 * L;
			Hobj = c1 * H * H + c2 * H;

			if (Lobj > Hobj+eps) a2 = L;
			else if (Lobj < Hobj-eps) a2 = H;
			else a2 = alpha2;
		}

		if (Math.abs(a2-alpha2) < eps*(a2+alpha2+eps))  return 0;

		a1 = alpha1 - s * (a2 - alpha2);
		if (a1 < 0) 
		{
			//System.out.println("clip a1 < 0");
			a2 += s * a1;
			a1 = 0;
		}
		else if (a1 > C) 
		{
			//System.out.println("clip a1 > C");
			float t = a1-C;
			a2 += s * t;
			a1 = C;
		}
		// update b----------------------------------
		float b1=0, b2=0, bnew=0;
  
		if (a1 > 0 && a1 < C)
			bnew = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
		else 
			{
				if (a2 > 0 && a2 < C)
					bnew = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
				else 
				{
					b1 = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
					b2 = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
					bnew = (b1 + b2) / 2;
				}
			}
		
		delta_b = bnew - b;
		b = bnew;
		//---------------------------------------------------
  
		if (kernel_type == linear_kernel_flag) 
		{
			float t1 = y1 * (a1 - alpha1);
			float t2 = y2 * (a2 - alpha2);

				for (int i=0; i<d; i++)
				{
					float temp = dense_points[i1][i] * t1 + dense_points[i2][i] * t2;;
					float temp1 = ((Float)w.get(i)).floatValue();
					Float value = new Float(temp + temp1);
					w.set(i,value);
				}
		}

		float t1 = y1 * (a1-alpha1);
		float t2 = y2 * (a2-alpha2);
		// update error cache
		for (int i=0; i<end_support_i; i++)
			if (0 < ((Float)alpha.get(i)).floatValue() && ((Float)alpha.get(i)).floatValue() < C)
			{  
					float tmp = ((Float)error_cache.elementAt(i)).floatValue();
					tmp +=  t1 * kernel_func(i1,i) + t2 * kernel_func(i2,i)- delta_b;
					error_cache.set(i,new Float(tmp));
			}
		error_cache.set(i1,new Float(0));
		error_cache.set(i2,new Float(0));

		//System.out.println("a1 "+a1+"-a2 "+a2);
		alpha.set(i1,new Float(a1));
		alpha.set(i2,new Float(a2));
		numStep++;
		
		return 1;
	}
	//------------------------------------------------------------
	float dot_product(int i1, int i2)
	{

		float dot = 0;
		for (int i=0; i<d; i++)
			dot += dense_points[i1][i] * dense_points[i2][i];
 
		return dot;
	}
	//--------------------------------------------
	float poly_kernel(int i1, int i2, int degree)
	{

		float res = (float)Math.pow(dot_product(i1,i2)+1,degree);
		//float res = (float)Math.pow(precomputed_dot_product[i1][i2],degree);
 
		return res;
	}

	//---------------------------------------------------------------
	float rbf_kernel(int i1, int i2)
	{
		float s = precomputed_dot_product[i1][i2]; // da duoc tinh san~ va luu
		s *= -2;
		s += precomputed_dot_product[i1][i1] + precomputed_dot_product[i2][i2];
		//s += ((Float)precomputed_self_dot_product.get(i1)).floatValue()
		//	+ ((Float)precomputed_self_dot_product.get(i2)).floatValue();
		return (float)Math.exp((float)(-s/two_sigma_squared));    
	}
	//----------------
	float kernel_func(int i1,int i2)
	{	
		float result =0;
		if (kernel_type == linear_kernel_flag)
			result = precomputed_dot_product[i1][i2];
			//result = dot_product(i1,i2);
		else if (kernel_type == poly_kernel_flag)
			result = poly_kernel(i1,i2,poly_degree);
		else if (kernel_type == rbf_kernel_flag)
			result = rbf_kernel(i1,i2);
		return result;
	}
	//--------------
	float learned_func_linear(int index)
	{

		float res = 0;
		for (int i=0; i<d; i++)
			res += ((Float)w.get(i)).floatValue() * dense_points[index][i];

		return (res-b);
	}
	// learned function nonlinear
	float learned_func(int index) 
	{
		if(kernel_type == linear_kernel_flag)return learned_func_linear(index);
		float res = 0;
		for (int i=0; i<end_support_i; i++)
			if (((Float)alpha.get(i)).floatValue() > 0)
			{   
				res += ((Float)alpha.get(i)).floatValue() * ((Integer)target.get(i)).floatValue() * kernel_func(i,index);			
			}
		return (res-b);
	}
	//-----------------
	float error_rate()
	{
		int total = 0;
		int error = 0;
		for (int i=0; i<N; i++) 
		{
			float res = learned_func(i);
			int tar = ((Integer)target.get(i)).intValue();
			//System.out.println(i+"-obj_func: "+res+"-tar: "+tar);
			//if((res>0 && tar <0)||(res<=0 && tar>0))
			if((res>=0 && tar <0)||(res<0 && tar>0))
				error++;
			total++;
		}
		return (float)error/(float)total;
	}
	
	public void train()
	{

		//if (!is_linear_kernel) 
		//{
			//my.resize(my.precomputed_self_dot_product,my.N,2);

			for (int i=0; i<N; i++)
				for (int j=0; j<N; j++)
					precomputed_dot_product[i][j] = dot_product(i,j);
//				{
//					if (i != j)
//						precomputed_dot_product[i][j] = dot_product(i,j);
//					else
//					{
//						float temp = dot_product(i,i);
//						//System.out.println(i+" "+j+" "+temp);
//						precomputed_self_dot_product.add(new Float(temp));
//						precomputed_dot_product[i][i] =temp;
//					}
//				}

		numChanged = 0;
		examineAll = 1;
		while (numChanged > 0 || examineAll >0) 
		{
			numChanged = 0;
			if (examineAll>0)    
				for (int k = 0; k < N; k++)
					numChanged += examineExample (k);
			else 
				for (int k = 0; k < N; k++)
					if (((Float)alpha.get(k)).floatValue() != 0 && ((Float)alpha.get(k)).floatValue() != C)
						numChanged += examineExample (k);
			if (examineAll == 1)
				examineAll = 0;
			else if (numChanged == 0)
				examineAll = 1;                    

		}
	}
	//*
	int read_data(String data_file_name) throws FileNotFoundException
	{
  
		FileInputStream  data = new FileInputStream(data_file_name);
		//DataInputStream is = new DataInputStream (data);
		 BufferedReader is = new BufferedReader(new InputStreamReader(data));

		String s = new String();
		int n_lines = 0;
		int n_attributes = 0;
		String attribute_string = new String("@attribute");
		String data_string = new String("@data");
		try
		{
		
			while(true)
			{
				s= is.readLine();
				if(s!=null)
				  if(s.length()>1)
					if(s.charAt(0)=='@')break;
			}

			while(true)
			{
				s= is.readLine();
				if(s!=null)
					if(s.length()>9)
						if(s.substring(0,10).equals(attribute_string))break;
			}
			while(s.length()>10 && s.substring(0,10).equals(attribute_string))
			{
				n_attributes++;
				s= is.readLine();
			}
			System.out.println("num of attributes: "+n_attributes);
			//--------------------
			d = n_attributes-1;
			while(true)
			{
				s= is.readLine();
				if(s!=null)
					if(s.length()>=5)
						if(s.substring(0,5).equals(data_string))break;
			}
			n_lines=0;
			s= is.readLine();
			//for ( n_lines=0; (s= is.readLine()) != null; n_lines++)
			while(s!=null)	
			{
				StringTokenizer st = new StringTokenizer(s,","); 
				Vector v =new Vector();
   
				//int g=0;
				try
				{
					while (st.hasMoreTokens()) 
					{
						//float tmp = Float.valueOf(st.nextToken()).floatValue();
						String tmp = st.nextToken();
						char c = tmp.charAt(0);
						String negative = new String("negative");
						String positive = new String("positive");

						if(c=='b')
							v.add(new Float(1));
						else if(c=='o')
							v.add(new Float(2));
						else if(c=='x')
							v.add(new Float(3));
						else if(tmp.equals(negative))
							v.add(new Float(-1));
						else if(tmp.equals(positive))
							v.add(new Float(1));
						else
						    v.add(Float.valueOf(tmp));
						//g++;
					}
				}
				catch (NumberFormatException e) 
				{
					System.err.println("Number format error " + e.toString());
				}
   
				int tar = (int)((Float)v.lastElement()).floatValue();
				if(tar ==0) tar = -1;  //----------------------------Mau chot day roi----------------
				target.add(new Integer (tar));      
				v.remove(v.size()-1);

				for ( int i=0; i<d; i++)
						dense_points[n_lines][i] = ((Float)v.get(i)).floatValue();			
				n_lines++;
				s= is.readLine();
			}
			is.close();
		}
		
		catch(Exception e)
		{
			e.printStackTrace();
		}
	
		N = n_lines;
		end_support_i=n_lines;
		for (int i=0; i<d; i++)
			w.add(new Float(0));
		// khoi tao alpha va error_cache
		for(int i=0;i<N;i++)
		{
			alpha.add(new Float(0));
			int e= ((Integer)target.get(i)).intValue()*(-1);
			error_cache.add(new Float(e));
		}
		return n_lines;
  
	}
	//*/
	//------------------------------------------------------------------------------------------
	// Generate my own data
	///*
	public void data_gen(int Num, int k)
	{
		if(k>MATRIX || Num > MATRIX)
		{
			System.out.println("out of array bound");
			System.exit(1);
		}
		//--------------------------------------------
		N = Num;
		d = k;
		end_support_i = N;
		w = new Vector();
		target = new Vector();
		alpha = new Vector();
		error_cache = new Vector();

		// ------------------tao du lieu moi-------------------------------------------------------
		for(int i=0;i<Num;i++)
		{
			for(int j =0; j<k;j++)
			{	
				float tmp= (float)Math.random();
				if(tmp<(float)0.5)
					dense_points[i][j] = 0;
				else
					dense_points[i][j] = 1;
			}
			// add label
			target.add(new Integer(compute_label_rule1(i,k)));
		}
		//---------------------------------------
		for (int i=0; i<d; i++)
			w.add(new Float(0));
		// khoi tao alpha va error_cache
		for(int i=0;i<N;i++)
		{
			alpha.add(new Float(0));
			int e= ((Integer)target.get(i)).intValue()*(-1);
			error_cache.add(new Float(e));
		}
	}
	//-------------------------------------------
	// rule to compute label
	int compute_label_rule1(int i,int k)
	{
		if(dense_points[i][0]==0)
			return -1;
		else
			return 1;
	}
	// rule to compute label
	int compute_label_rule2(int i,int k)
	{
		int tmp = 0;
		for(int j =0; j<k;j++)
			tmp += dense_points[i][j];
		
		if(((k%2==0) && tmp==(k/2))|| tmp>(k/2))
			return 1;
		else
			return -1;
	}// */
	//----------------------------------------------------------------------------------
	///*
	public void write_mydata(String filename) throws FileNotFoundException
	{
		 FileOutputStream  data = new FileOutputStream(filename);
		 PrintStream os = new PrintStream(data);
		 //DataInputStream is = new DataInputStream (data);
		 //BufferedWriter os = new BufferedWriter(new OutputStreamWriter(data));
		 try{

			 os.println(N);
			 os.println(d);
			 for(int i =0; i<N; i++ )
			 {
				 for(int j =0; j<d; j++)
				 {
					 os.print((int)dense_points[i][j]);
					 os.print(',');
				 }
				 os.println(((Integer)target.get(i)).intValue());
			 }
			 os.close();
		 }
		 catch(Exception e)
		 {
			 
		 }
		 
	}// */
	//
	public void read_mydata(String filename) throws FileNotFoundException
	{
		FileInputStream  data = new FileInputStream(filename);

		BufferedReader is = new BufferedReader(new InputStreamReader(data));

		String s = new String();
		 try
		 {
			 s = is.readLine();
			 N = Integer.valueOf(s);
			 s = is.readLine();
			 d = Integer.valueOf(s);
			 end_support_i = N;
			 w = new Vector();
			 target = new Vector();
			 alpha = new Vector();
			 error_cache = new Vector();

			 for(int i =0; i< N; i++)	
			 {
				    s = is.readLine();
				 	StringTokenizer st = new StringTokenizer(s,","); 
					Vector v =new Vector();

					try
					{
						while (st.hasMoreTokens()) 
						{
							//float tmp = Float.valueOf(st.nextToken()).floatValue();
							String tmp = st.nextToken();
							    v.add(Float.valueOf(tmp));
						}
					}
					catch (NumberFormatException e) 
					{
						System.err.println("Number format error " + e.toString());
					}
	   
					int tar = (int)((Float)v.lastElement()).floatValue();
					if(tar == 0) tar = -1;  //----------------------------Mau chot day roi----------------
					target.add(new Integer (tar));      
					v.remove(v.size()-1);

					for ( int j=0; j < d; j++)
							dense_points[i][j] = ((Float)v.get(j)).floatValue();
			}
				
			//------------------------------
			for (int i=0; i<d; i++)
					w.add(new Float(0));
			// khoi tao alpha va error_cache
			for(int i=0;i<N;i++)
			{
					alpha.add(new Float(0));
					int e= ((Integer)target.get(i)).intValue()*(-1);
					error_cache.add(new Float(e));
			}
			is.close();
		 }
		 catch(Exception e)
		 {
			 e.printStackTrace();
		 }

	}
	// 11-4
	public void increase_data(int delta)
	{
		for(int i=0;i<N;i++)
			for(int j=0;j<d;j++)
				dense_points[i][j] +=delta;
	}
	public void multiple_data(int times)
	{
		for(int i=0;i<N;i++)
			for(int j=0;j<d;j++)
				dense_points[i][j] *=times;
	}
	public void flip_label()
	{
		int tmp;
		for(int i=0;i<N;i++)
		{
			tmp =  ((Integer)target.get(i)).intValue()*(-1);
			target.set(i,new Integer (tmp));
		}
	}
	//------------------------------------------------------------------------------------------
	public static void main(String args[]) throws Exception
	{
		SMO mySMO = new SMO();
		//mySMO.kernel_type = mySMO.poly_kernel_flag;
		//mySMO.kernel_type = mySMO.linear_kernel_flag;
		mySMO.poly_degree = 3;
		mySMO.C = (float)3000;

//		mySMO.data_gen(8, 6);
//		mySMO.write_mydata("data14.txt");
		
		mySMO.read_mydata("data16.txt");
		//mySMO.flip_label();
		System.out.println(mySMO.N+" "+mySMO.d+" ");
		//mySMO.increase_data(3);
		//mySMO.multiple_data(-1);
		mySMO.train();
		for(int i = 0;i<mySMO.N;i++)
			System.out.println("alpha "+i+" : "+mySMO.alpha.get(i));
		for(int i = 0;i<mySMO.d;i++)
			System.out.println("w "+i+" : "+mySMO.w.get(i));
		//System.out.println("Num of TakeStep: "+mySMO.numStep);
		System.out.println("b: "+mySMO.b);
		float error = mySMO.error_rate();
		System.out.println("error rate: "+error);
	}
}
