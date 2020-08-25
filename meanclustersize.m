clear;
%T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/ABP/noname2/xyposition8000.txt.txt",'ReadVariableNames',0);
x = linspace(0,20000,201)';
T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/exp_d0.5vcorre.txt",'ReadVariableNames',0);
x1=T1.Var1;
y1=T1.Var2; 
T2=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/exp_d0.1vcorre.txt",'ReadVariableNames',0);
x2=T2.Var2;
y2=T2.Var2; 
%y=[y1 y2 y3];
%ymean = mean(y,2);
%ystd = std(y,0,2);

%y2=[y4 y5 y6];
%ymean2 = mean(y2,2);

figure;
set(gca,'FontSize',30);
axis square;
box on;
axis([0,300,-1,1])
set(gca,'linewidth',2);
hold on;
plot(x1,y1,'r-o','LineWidth',2,'MarkerFaceColor','r');
%errorbar(x,ymean,ystd,'-or')
plot(x1,y2,'k-o','LineWidth',2,'MarkerFaceColor','k')

%ipt = findchangepts(x, 'Statistic','std');
%b = polyfit(x(1:ipt), ymean(1:ipt), 1);
%Yf = polyval(b, x(1:ipt));
%plot(x(1:ipt), Yf, ':r', 'LineWidth',2)
legend({'Experiment \rho=10^{-4} \mu m^{-2}','Experiment \rho=5 \times 10^{-4} \mu m^{-2}' },'FontSize',15,'TextColor','black')
%text(0.2, 0.8, sprintf('y = %6.4f x + %6.3f', b),'FontSize',20 )

%title('Velocity Correlation','FontSize',20)
%xlabel('Distance (\mum)') 
xlabel('r(\mum)','FontSize',40) 
ylabel('C(r)','FontSize',40,'Rotation',360,'Position',[-70,0]) 





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




