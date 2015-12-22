var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = 2 * (figureWidth + figureMargin.left + figureMargin.right);  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = 2 * (figureWidth + figureMargin.top + figureMargin.bottom);  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 10};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Read the data in.
var dataAccessorFunction = function(d)
    {
        return {First : +d.FirstModel, Second : +d.SecondModel};
    }
d3.tsv("/Data/DisambiguationResults.tsv", dataAccessorFunction, function(error, data)
    {
        // Sort the data by the value of the first model.
        data = data.sort(function(a, b) { return d3.ascending(a.First, b.First); });

        // Create the figure the type 1 vs type 2 diabetes results.
        var figure1v2 = svg.append("g")
            .attr("transform", "translate(0, 0)");
//REMOVE
        figure1v2.append("rect").attr("height", figureHeight).attr("width", figureWidth).style("opacity", 0.1);
//REMOVE
        createFigure(figure1v2, data.map(function(d, index) { return {First : d.First, Second : d.Second, index : index}; }), "Type 1 Vs. Type 2 Diabetes");

        // Create the figure for the diabetes vs non-diabetes results.
        var figureDvND = svg.append("g")
            .attr("transform", "translate(" + (svgWidth / 2) + ", 0)");
//REMOVE
        figureDvND.append("rect").attr("height", figureHeight).attr("width", figureWidth).style("opacity", 0.1);
//REMOVE
        createFigure(figureDvND, data.map(function(d, index) { return {First : d.First, Second : d.Second, index : index}; }), "Diabetes Vs. Non-diabetes");
    }
);

function createFigure(figureContainer, dataArray, figureTitle)
{
    // Define positioning variables.
    var figurePadding = {top: 40, right: 5, bottom: 50, left: 60};  // Padding around the graph in the figure to allow for titles and labelling axes.

    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0, dataArray.length + 1])  // Add 1 to allow for the final datapoint to have space for a line.
        .range([figurePadding.left, figureWidth - figurePadding.right]);
    var yScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([figureHeight - figurePadding.bottom, figurePadding.top]);

    // Add the axes for the figure.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - figurePadding.bottom) + ")")
        .call(xAxis);
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    yAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + figurePadding.left + ", 0)")
        .call(yAxis);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figurePadding.top * 3 / 4)
        .text(figureTitle);

    // Add the labels on the axes. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
    // to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
    var xAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figureHeight - (figurePadding.bottom / 4))
        .text("Sample Index");
    yAxisLabelLoc = { x : figurePadding.left / 3, y : ((figureHeight - figurePadding.top - figurePadding.bottom) / 2) + figurePadding.top };
    var yAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", yAxisLabelLoc.x)
        .attr("y", yAxisLabelLoc.y)
        .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
        .text("Posterior Probability");

    // Add the lines for the data.
    var dataPathFirstModel = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d)
    {
        dataPathFirstModel += "L" + xScale(d.index) + "," + yScale(d.First) + "h1";
    });
//    dataPathFirstModel += "V" + yScale(0) + "H" + xScale(0);
    var dataLineFirstModel = figureContainer.append("path")
        .attr("class", "data firstModel")
        .attr("d", dataPathFirstModel);
    var dataPathSecondModel = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d)
    {
        dataPathSecondModel += "L" + xScale(d.index) + "," + yScale(d.Second) + "h1";
    });
//    dataPathSecondModel += "V" + yScale(0) + "H" + xScale(0);
    var dataLineSecondModel = figureContainer.append("path")
        .attr("class", "data secondModel")
        .attr("d", dataPathSecondModel);
}