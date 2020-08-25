clear;
%T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/ABP/noname2/xyposition8000.txt.txt",'ReadVariableNames',0);
x = linspace(0,20000,201)';
T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/group1/data5/xyposition0.txt.txt" ,'ReadVariableNames',0);
x1=T1.Var1; y1=T1.Var2; 
T2=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/group1/data5/xyposition6000.txt.txt",'ReadVariableNames',0);
x2=T2.Var1; y2=T2.Var2; 
T3=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/group1/data5/xyposition12000.txt.txt",'ReadVariableNames',0);
x3=T3.Var1; y3=T3.Var2;
T4=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/group1/data5/xyposition18000.txt.txt",'ReadVariableNames',0);
x4=T4.Var1; y4=T4.Var2; 
T5=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation64/density0.5/ABP2vc.txt",'ReadVariableNames',0);
x5=T5.Var1; y5=T5.Var2; 
T6=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation64/density0.5/ABP3vc.txt",'ReadVariableNames',0);
x6=T6.Var1; y6=T6.Var2; 

%y=[y1 y2 y3];
%ymean = mean(y,2);
%ystd = std(y,0,2);

%y2=[y4 y5 y6];
%ymean2 = mean(y2,2);


figure;
set(gca,'FontSize',30);
axis square;
box on;
axis([0,0.7,-0.5,0.5])
set(gca,'linewidth',2);
hold on;
plot(x1(2:end),y1(2:end),'b-o','LineWidth',2,'MarkerFaceColor','b');
%errorbar(x1,ymean,ystd,'-or')
plot(x2(2:end),y2(2:end),'k-o','LineWidth',2,'MarkerFaceColor','k')
plot(x3(2:end),y3(2:end),'g-o','LineWidth',2,'MarkerFaceColor','g')
plot(x4(2:end),y4(2:end),'r-o','LineWidth',2,'MarkerFaceColor','r')
legend({'Step 0','Step 6000','Step 12000','Step 18000' },'FontSize',20,'TextColor','black')

%ipt = findchangepts(x, 'Statistic','std');
%b = polyfit(x(1:ipt), ymean(1:ipt), 1);
%Yf = polyval(b, x(1:ipt));
%plot(x(1:ipt), Yf, ':r', 'LineWidth',2)
%legend({'APPA','Classic ABP' ,'Linear Fit'},'FontSize',12,'TextColor','black')
%text(0.2, 0.8, sprintf('y = %6.4f x + %6.3f', b),'FontSize',20 )

ylabel('C(r)','FontSize',40,'Rotation',360)
xlabel('r','FontSize',40) 
%xlabel('Time Step','FontSize',20) 
%ylabel('Normalized Cluster Size','FontSize',20) 




%{
figure;
axis square;
box on;
axis([0,360,-1,1])
set(gca,'linewidth',2);
hold on;
plot(x1,y1,'k-o','LineWidth',3,'MarkerFaceColor','k')
plot(x2,y2,'r-o','LineWidth',3,'MarkerFaceColor','r')
legend({'Simulation Density0.1','Simulation Density0.5','Experiment Density0.1','Experiment Density0.5'},'FontSize',12,'TextColor','blue')
title('Velocity Correlation','FontSize',20)
%}




