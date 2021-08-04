library(shiny)
library(plotly)
library(tibble)
library(dplyr)

source("main.R")

x1 <- seq(0,2*pi, length.out = 500)
y1 <- sin(x1)
y2 <- sin(x1)+0.25
y3<- sin(x1)+0.5

data1 <- tibble(x=x1,y=y1,NN=y2)
data1 %>% 
  mutate(NN=y2)

countries <- c( # may be edited
  "World", "Germany", "France", "Austria", 
  "Belgium", "Italy", "Portugal", "Spain")

plot_param <- readr::read_csv("plot_param.csv")

ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      sliderInput("anzahl", "Anzahl hidden Layer", 
                   1, 10, 3, step=1, sep=""),
      sliderInput("breite", "Breite hidden Layer", 
                  1, 50, 30, step=1, sep=""),
      radioButtons("iterations", "Anzahl der Iterations",
                   choices=list("10"=10,"500"=500,"1000"=1000,"5000"=5000), 
                   selected=10, inline=TRUE),
      radioButtons("percentile", "Confidence", 
                   choices=list("33%"=33,"50%"=50,"67%"=67), 
                   selected=67, inline=TRUE)
    ),
    mainPanel(
      plotlyOutput("plot")
    )))

server <- function(input, output) {
  #output$trend_ui <- renderUI({
  #   if (input$show_trend) 
  #     sliderInput("trend_range", "Trend from Range", 
  #                 1900, 2019, c(2000, 2018), step=1, sep="")
  # })
  output$plot <- renderPlotly({
    a <- rep(input$breite,input$anzahl)
    print(a)
    N1 <- NN$new(length(a),c(1,a,1))
    N1$GD3(x1,y1,delta=0.01,iteration = input$iterations)
    y3 <- N1$calculate2(x1)
    data1 %>%
      mutate(NN=y3) ->
      data1
    plt <- ggplot(data = data1, aes(x = x1))+ geom_line(aes(y=y1),show.legend = TRUE) + geom_line(aes(y=NN),color="red",show.legend = TRUE)
    ggplotly(plt)
  })
}

shinyApp(ui, server)
