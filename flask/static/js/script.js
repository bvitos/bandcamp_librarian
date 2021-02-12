function checkifdone() {
    $.ajax({
            url: "/warning_message",
            type: "POST",
            success: function(response) {
                $("#wrn").text(response.message);
                if (response.message == "") {
                    document.getElementById("button1").disabled = false
                };
            }
        })
}


function clickmessage() {
    $("#wrn").text("Processing, please wait!");
    document.getElementById("button1").disabled = true;
    var myVar = setInterval(checkifdone, 1000);
}