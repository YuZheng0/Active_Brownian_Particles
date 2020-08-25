T1=readtable("/Users/yuzheng/Documents/Research_data/ABP_data/Revisedpaper data/rotation256/alpha/density0.5/data2/position19990.txt");
x0=T1.Var1;

y0=T1.Var2; 

vx = T1.Var3;
vy =T1.Var4;
figure;
scatter(x0,y0,190,'MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor','r','LineWidth',1.5);
axis square;
box on;
set(gca,'linewidth',3);
axis([0,1,0,1])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
hold on;
%q = quiver(x0,y0,vx,vy,'-r','LineWidth',2.5,'MaxHeadSize',10,'AutoScaleFactor',1,'AutoScale','on');
