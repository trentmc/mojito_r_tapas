while 1
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*best_team_nmse=//' -e 's/;.*//'> best_team_nmses.txt"); 
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*best_ind_nmse=//' -e 's/;.*//'> best_ind_nmses.txt"); 
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*avg_team_nmse=//' -e 's/;.*//'> avg_team_nmses.txt"); 
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*avg_ind_nmse=//' -e 's/;.*//'> avg_ind_nmses.txt"); 
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*avg_team_age=//' -e 's/;.*//'> avg_team_ages.txt"); 
  system("cat out_andes.txt |grep 'Gen ' | sed -e 's/.*avg_ind_age=//' -e 's/;.*//'> avg_ind_ages.txt"); 
  
  load best_team_nmses.txt; 
  load best_ind_nmses.txt;
  load avg_team_nmses.txt;
  load avg_ind_nmses.txt;
  load avg_team_ages.txt;
  load avg_ind_ages.txt;
  
  y1 = best_team_nmses;
  y2 = best_ind_nmses;
  
  %y2 = avg_team_nmses;
  
  %y1 = best_ind_nmses;
  %y2 = avg_ind_nmses;
  x = 1:length(y1);
  plot(x, y1, 'r', x, y2, 'g');
  pause(0.5)
end
