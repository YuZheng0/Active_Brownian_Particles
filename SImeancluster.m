clear;
%T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/ABP/noname2/xyposition8000.txt.txt",'ReadVariableNames',0);
x = linspace(0,20000,201)';
T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/density0.5/2mean_cluster_numbers.txt",'ReadVariableNames',0);
y1=T1.Var1;

%T2=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/alpha/data1/mean_cluster_numbers.txt",'ReadVariableNames',0);
%y2=T2.Var1;

figure;
set(gca,'FontSize',30);
axis square;
box on;
axis([0,200,0,1])
set(gca,'linewidth',2);
hold on;
plot(y1,'r-o','LineWidth',2,'MarkerFaceColor','r');
xlabel('t','FontSize',40) 
ylabel('S','FontSize',40,'Rotation',360,'Position',[-40,0.5]) 