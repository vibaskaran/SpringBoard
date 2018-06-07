function getData(){
    var xhttp = new XMLHttpRequest();

    xhttp.onreadystatechange = function(){
        if(this.readyState == 4 && this.status == 200){
           alert(this.responseText);
        }
    }

    var dropdown = document.getElementById("myDropdown");//select the dropdown
    var strUser = dropdown.options[dropdown.selectedIndex].text; //get the selected value
    console.log(strUser);
    xhttp.open("GET", strUser, true);
    xhttp.send(strUser);
  }
